package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/inceptionstack/bedrockify"
)

var (
	version = "dev"
	commit  = "none"
	date    = "unknown"
)

func main() {
	port := flag.Int("port", 0, "Port to listen on (default 8090)")
	host := flag.String("host", "", "Host to bind to (default \"127.0.0.1\")")
	region := flag.String("region", "", "AWS region for Bedrock (default \"us-east-1\")")
	baseURL := flag.String("base-url", "", "Custom Bedrock endpoint URL (overrides region-based endpoint)")
	model := flag.String("model", "", "Default Bedrock model ID")
	embedModel := flag.String("embed-model", "", "Default Bedrock embedding model ID (default \"amazon.titan-embed-text-v2:0\")")
	bearerToken := flag.String("bearer-token", "", "Bedrock API key (bearer token)")
	initConfig := flag.Bool("init", false, "Write example bedrockify.toml to current directory and exit")
	showVersion := flag.Bool("version", false, "Show version and exit")
	flag.Parse()

	if *showVersion {
		fmt.Printf("bedrockify %s (commit: %s, built: %s)\n", version, commit, date)
		os.Exit(0)
	}

	if *initConfig {
		path := "bedrockify.toml"
		if err := bedrockify.WriteExampleConfig(path); err != nil {
			log.Fatalf("Failed to write config: %v", err)
		}
		fmt.Printf("✓ Wrote %s\n", path)
		os.Exit(0)
	}

	// Load config: defaults → file → env vars
	cfg, loadedFrom, err := bedrockify.LoadConfig()
	if err != nil {
		log.Fatalf("Config error: %v", err)
	}
	if loadedFrom != "" {
		log.Printf("Loaded config from %s", loadedFrom)
	}

	// CLI flags override config (only if explicitly set)
	if *port != 0 {
		cfg.Port = *port
	}
	if *host != "" {
		cfg.Host = *host
	}
	if *region != "" {
		cfg.Region = *region
	}
	if *baseURL != "" {
		cfg.BaseURL = *baseURL
	}
	if *model != "" {
		cfg.Model = *model
	}
	if *embedModel != "" {
		cfg.EmbedModel = *embedModel
	}
	if *bearerToken != "" {
		cfg.BearerToken = *bearerToken
	}

	// Handle subcommands
	if args := flag.Args(); len(args) > 0 {
		switch args[0] {
		case "install-daemon":
			if err := runInstallDaemon(cfg); err != nil {
				log.Fatalf("Install daemon failed: %v", err)
			}
			os.Exit(0)
		case "update":
			if err := runUpdate(version); err != nil {
				log.Fatalf("Update failed: %v", err)
			}
			os.Exit(0)
		default:
			fmt.Fprintf(os.Stderr, "Unknown command: %s. Available: install-daemon, update, --init\n", args[0])
			os.Exit(1)
		}
	}

	// Create chat converser
	var conv bedrockify.Converser
	if cfg.BearerToken != "" {
		log.Printf("Using Bedrock API key (bearer token) authentication")
		conv, err = bedrockify.NewBedrockConverserWithBearerToken(cfg.Region, cfg.Model, cfg.BearerToken, cfg.BaseURL)
	} else {
		conv, err = bedrockify.NewBedrockConverser(cfg.Region, cfg.Model, cfg.BaseURL)
	}
	if err != nil {
		log.Fatalf("Failed to create converser: %v", err)
	}

	// Create embedder (optional)
	var emb bedrockify.Embedder
	if cfg.EmbedModel != "" {
		emb, err = bedrockify.NewBedrockEmbedder(cfg.Region, cfg.EmbedModel)
		if err != nil {
			log.Fatalf("Failed to create embedder: %v", err)
		}
	}

	handler := bedrockify.NewHandlerFull(conv, emb, cfg.Model, cfg.EmbedModel, version, cfg.Region)
	addr := fmt.Sprintf("%s:%d", cfg.Host, cfg.Port)

	endpointInfo := ""
	if cfg.BaseURL != "" {
		endpointInfo = ", endpoint=" + cfg.BaseURL
	}
	embedInfo := ""
	if cfg.EmbedModel != "" {
		embedInfo = ", embed=" + cfg.EmbedModel
	}
	log.Printf("bedrockify %s starting on http://%s (region=%s, model=%s%s%s)", version, addr, cfg.Region, cfg.Model, embedInfo, endpointInfo)

	server := &http.Server{
		Addr:              addr,
		Handler:           handler,
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       30 * time.Second,
		WriteTimeout:      300 * time.Second,
		IdleTimeout:       120 * time.Second,
	}
	if err := server.ListenAndServe(); err != nil {
		log.Fatalf("Server error: %v", err)
	}
}
