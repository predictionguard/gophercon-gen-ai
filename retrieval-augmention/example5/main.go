package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"strings"

	cohere "github.com/cohere-ai/cohere-go"
)

// Define the API details to access the LLM.
var url = "https://api.predictionguard.com/completions"

// Stop tokens.
var stop []string = []string{
	"#",
	"import",
	"Human",
	"human",
	"AI:",
}

// CompletionResult is a single completion result.
type CompletionResult struct {
	Text   string      `json:"text"`
	Output interface{} `json:"output,omitempty"`
	Index  int         `json:"index"`
	Status string      `json:"status"`
	Model  string      `json:"model"`
}

// CompletionResults is a list of completion results.
type CompletionResults struct {
	Id      string             `json:"id"`
	Object  string             `json:"object"`
	Created int64              `json:"created"`
	Choices []CompletionResult `json:"choices"`
}

// TypedOutput is a struct type that represents a typed output.
type TypedOutput struct {
	Type        string   `json:"type"`
	Categories  []string `json:"categories"`
	Pattern     string   `json:"pattern"`
	Consistency bool     `json:"consistency"`
	Factuality  bool     `json:"factuality"`
	Toxicity    bool     `json:"toxicity"`
}

// CompletionRequest is a struct type that represents a completion request.
type CompletionRequest struct {
	Model       string      `json:"model"`
	Prompt      string      `json:"prompt"`
	MaxTokens   int         `json:"max_tokens"`
	Temperature float64     `json:"temperature"`
	Output      TypedOutput `json:"output"`
}

// getCompletions calls the Prediction Guard API to get text completions.
func getCompletions(request CompletionRequest) (*CompletionResults, error) {

	// Prepare the payload.
	payload, err := json.Marshal(request)
	if err != nil {
		return nil, err
	}

	// Make the POST request.
	client := &http.Client{}
	req, err := http.NewRequest("POST", url, strings.NewReader(string(payload)))
	if err != nil {
		return nil, err
	}
	req.Header.Add("Authorization", "Bearer "+os.Getenv("PREDICTIONGUARD_TOKEN"))
	req.Header.Add("Content-Type", "application/json")
	res, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()

	// Read the response body into a CompletionResults value.
	var results CompletionResults
	if err := json.NewDecoder(res.Body).Decode(&results); err != nil {
		return nil, err
	}

	return &results, nil
}

// qAPromptTemplate is a template for a question and answer prompt.
func qAPromptTemplate(context, question string) string {
	return fmt.Sprintf(`### Instruction:
Read the context below and answer the question. If the question cannot be answered based on the context alone or the context does not explicitly say the answer to the question, respond "Sorry I had trouble answering this question, based on the information I found."

### Input:
Context: "%s"

Question: "%s"
	
### Repsonse:
`, context, question)
}

// embed vectorizes a user message/query.
func embed(message string, co *cohere.Client) ([]float64, error) {
	res, err := co.Embed(cohere.EmbedOptions{
		Model: "embed-english-light-v2.0",
		Texts: []string{message},
	})
	if err != nil {
		return nil, err
	}
	return res.Embeddings[0], nil
}

// VectorizedChunk is a struct that holds a vectorized chunk.
type VectorizedChunk struct {
	Chunk  string    `json:"chunk"`
	Vector []float64 `json:"vector"`
}

// VectorizedChunks is a slice of vectorized chunks.
type VectorizedChunks []VectorizedChunk

// cosineSimilarity calculates the cosine similarity between two vectors.
func cosineSimilarity(a []float64, b []float64) (cosine float64, err error) {
	count := 0
	length_a := len(a)
	length_b := len(b)
	if length_a > length_b {
		count = length_a
	} else {
		count = length_b
	}
	sumA := 0.0
	s1 := 0.0
	s2 := 0.0
	for k := 0; k < count; k++ {
		if k >= length_a {
			s2 += math.Pow(b[k], 2)
			continue
		}
		if k >= length_b {
			s1 += math.Pow(a[k], 2)
			continue
		}
		sumA += a[k] * b[k]
		s1 += math.Pow(a[k], 2)
		s2 += math.Pow(b[k], 2)
	}
	if s1 == 0 || s2 == 0 {
		return 0.0, errors.New("vectors should not be null (all zeros)")
	}
	return sumA / (math.Sqrt(s1) * math.Sqrt(s2)), nil
}

// search through the vectorized chunks to find the most similar chunk.
func search(chunks VectorizedChunks, embedding []float64) (string, error) {
	outChunk := ""
	var maxSimilarity float64 = 0.0
	for _, c := range chunks {
		distance, err := cosineSimilarity(c.Vector, embedding)
		if err != nil {
			return "", err
		}
		if distance > maxSimilarity {
			outChunk = c.Chunk
			maxSimilarity = distance
		}
	}
	return outChunk, nil
}

// getRAGAnswer gets a retrieval based answer.
func getRAGAnswer(input string, chunks VectorizedChunks, co *cohere.Client) (string, error) {

	// Embed a question for the RAG answer.
	embedding, err := embed(input, co)
	if err != nil {
		return "", err
	}

	// Search for the relevant chunk.
	chunk, err := search(chunks, embedding)
	if err != nil {
		return "", err
	}

	// Prompt with the Q&A template.
	request := CompletionRequest{
		Prompt: qAPromptTemplate(
			chunk,
			input,
		),
		Model: "Nous-Hermes-Llama2-13B",
	}
	response, err := getCompletions(request)
	if err != nil {
		return "", err
	}

	// Post process the completion. Given that we are using a system prompt,
	// Llama 2 models might return some "extra" stuff after the input/output indicators
	// so we will truncate the completion string on the first # encountered.
	completion := string(response.Choices[0].Text)
	for _, s := range stop {
		if strings.Contains(completion, s) {
			completion = completion[:strings.Index(completion, s)]
		}
	}
	completion = strings.TrimSpace(completion)

	return completion, nil
}

// chatContext is a struct that holds a chat context.
type chatContext struct {
	You string
	AI  string
}

// chatContexts is a slice of chat contexts.
type chatContexts []chatContext

// getChatAnswer gets a non-informational chat based answer.
func getChatAnswer(input string, context chatContexts, co *cohere.Client) (string, error) {

	// Take the last three chat contexts and format them into a string.
	beg := len(context) - 3
	if beg < 0 {
		beg = 0
	}
	filteredContext := context[beg:]
	filteredContextString := ""
	for _, fc := range filteredContext {
		filteredContextString += "Human: " + fc.You + "\nAI: " + fc.AI + "\n\n"
	}

	// Prompt with the chat context.
	request := CompletionRequest{
		Prompt: fmt.Sprintf(`### Instruction:
You are a helpful and kind chat assistant. Respond to the below user input based on the following conversation context:

%s
### Input: 
%s

### Response:
`, filteredContextString, input),
		Model: "WizardCoder",
		Output: TypedOutput{
			Type:       "categorical",
			Categories: []string{"yes", "no"},
		},
	}
	response, err := getCompletions(request)
	if err != nil {
		return "", err
	}

	return response.Choices[0].Text, nil
}

func main() {

	// Connect to Cohere.
	apiKey := os.Getenv("COHERE_API_KEY")
	if apiKey == "" {
		fmt.Fprintln(os.Stderr, "COHERE_API_KEY not specified")
		os.Exit(1)
	}
	co, err := cohere.CreateClient(apiKey)
	if err != nil {
		log.Fatal(err)
	}

	// Open the JSON file and load in the vectorized embeddings.
	f, err := os.Open("../example2/chunks.json")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	// Load it into the chunks value.
	var chunks VectorizedChunks
	if err := json.NewDecoder(f).Decode(&chunks); err != nil {
		log.Fatal(err)
	}

	// Start a cycle of listening for questions and responding to the questions.
	fmt.Println("")
	convo := chatContexts{}
	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("🧑: ")
		if !scanner.Scan() {
			break
		}
		input := scanner.Text()

		// Exit if we type "exit".
		if strings.ToLower(input) == "exit" {
			break
		}

		// Determine if the user is making an inquiry or just wants to chat.
		request := CompletionRequest{
			Prompt: fmt.Sprintf(`### Instruction:
Is the user asking an informational question or just wanting to chat? Answer "yes" if they are asking an informational question.

### Input: 
%s

### Response:
`, input),
			Model: "Nous-Hermes-Llama2-13B",
			Output: TypedOutput{
				Type:       "categorical",
				Categories: []string{"yes", "no"},
			},
		}
		response, err := getCompletions(request)
		if err != nil {
			log.Fatal(err)
		}

		// Handle the input accordingly.
		var completion string
		switch response.Choices[0].Text {
		case "yes":
			completion, err = getRAGAnswer(input, chunks, co)
			if err != nil {
				log.Fatal(err)
			}
		default:
			completion, err = getChatAnswer(input, convo, co)
			if err != nil {
				log.Fatal(err)
			}
		}

		fmt.Print("\n🤖: " + completion + "\n\n")

		// Add the chat context to the slice.
		convo = append(convo, chatContext{
			You: input,
			AI:  completion,
		})
	}
}
