package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
)

// Define the API details to access the LLM.
var url = "https://api.predictionguard.com/completions"

// Stop tokens.
var stop []string = []string{
	"#",
	"import",
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

func main() {

	// Get the prompt file from a command line argument.
	if len(os.Args) < 2 {
		log.Fatal("Please provide a prompt file as an argument.")
	}
	promptFile := os.Args[1]

	// Read in the prompt from a file.
	prompt, err := os.ReadFile(promptFile)
	if err != nil {
		log.Fatal(err)
	}

	// Prompt a Code Generation LLM.
	request := CompletionRequest{
		Prompt: string(prompt),
		Model:  "Nous-Hermes-Llama2-13B",
		Output: TypedOutput{
			Toxicity: true,
			//Factuality: true,
		},
	}
	response, err := getCompletions(request)
	if err != nil {
		log.Fatal(err)
	}
	if response.Choices[0].Status != "success" {
		log.Fatal(response.Choices[0].Status)
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

	// Print the autocompletion.
	fmt.Println("\n" + string(prompt) + completion)
}
