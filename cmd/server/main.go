package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/jacek/fluxgate/internal/server"
	"github.com/jacek/fluxgate/pkg/config"
)

func main() {
	configPath := flag.String("config", "config.yaml", "path to config file")
	flag.Parse()

	cfg, err := config.Load(*configPath)
	if err != nil {
		log.Fatalf("failed to load config: %v", err)
	}

	// Set defaults from environment if config is empty
	if len(cfg.Providers) == 0 {
		cfg.Providers = loadProvidersFromEnv()
	}

	if err := cfg.Validate(); err != nil {
		log.Fatalf("invalid config: %v", err)
	}

	srv, err := server.New(cfg)
	if err != nil {
		log.Fatalf("failed to create server: %v", err)
	}

	fmt.Printf("FluxGate starting on %s\n", srv.ListenAddr())
	if err := srv.Start(); err != nil {
		log.Fatalf("server error: %v", err)
	}
}

func loadProvidersFromEnv() []config.ProviderConfig {
	var providers []config.ProviderConfig

	// OpenAI provider from env
	if apiKey := os.Getenv("OPENAI_API_KEY"); apiKey != "" {
		providers = append(providers, config.ProviderConfig{
			Name:    "openai",
			Type:    "openai",
			BaseURL: getEnvOrDefault("OPENAI_BASE_URL", "https://api.openai.com"),
			APIKeys: []string{apiKey},
			Weight:  1,
		})
	}

	// Anthropic provider from env
	if apiKey := os.Getenv("ANTHROPIC_API_KEY"); apiKey != "" {
		providers = append(providers, config.ProviderConfig{
			Name:    "anthropic",
			Type:    "anthropic",
			BaseURL: getEnvOrDefault("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
			APIKeys: []string{apiKey},
			Weight:  1,
		})
	}

	return providers
}

func getEnvOrDefault(key, defaultVal string) string {
	if val := os.Getenv(key); val != "" {
		return val
	}
	return defaultVal
}
