package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"time"
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

func main() {

	// Define our prompt.
	prompt := "A great name for a unknown wizard (other than Gandalf and Radagast) from the Lord of the Rings universe is "

	// Loop over a range of temperature values from 0.1 to 2.0,
	// in increments of 0.4.
	for temp := 0.1; temp <= 2.0; temp += 0.4 {

		fmt.Println("\nTemperature:", temp)
		fmt.Println("--------------------------")

		// Generate three completions for this temperature value.
		for i := 0; i < 3; i++ {

			// Prompt the LLM with the temperature value.
			request := CompletionRequest{
				Prompt:      prompt,
				Model:       "Camel-5B",
				Temperature: temp,
				MaxTokens:   20,
			}
			response, err := getCompletions(request)
			if err != nil {
				log.Fatal(err)
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
			fmt.Println(completion)

			// Sleep for a second for rate limiting.
			time.Sleep(time.Second)
		}
	}
}
