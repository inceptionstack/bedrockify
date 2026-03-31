package main

import (
	"fmt"
	"os"
	"path/filepath"
	"text/template"

	"github.com/inceptionstack/bedrockify"
)

const daemonTemplate = `[Unit]
Description=bedrockify - OpenAI-compatible proxy for Amazon Bedrock
After=network.target

[Service]
Type=simple
ExecStart={{.ExecPath}} --port {{.Port}} --host {{.Host}} --region {{.Region}} --model {{.Model}}{{if .EmbedModel}} --embed-model {{.EmbedModel}}{{end}}{{if .BearerToken}}
Environment=AWS_BEARER_TOKEN_BEDROCK={{.BearerToken}}{{end}}{{if .BaseURL}}
Environment=BEDROCKIFY_BASE_URL={{.BaseURL}}{{end}}
Restart=on-failure
RestartSec=5s
StandardOutput=journal
StandardError=journal
SyslogIdentifier=bedrockify

[Install]
WantedBy=multi-user.target
`

type daemonVars struct {
	ExecPath    string
	Port        int
	Host        string
	Region      string
	Model       string
	EmbedModel  string
	BearerToken string
	BaseURL     string
}

func runInstallDaemon(cfg bedrockify.Config) error {
	execPath, err := os.Executable()
	if err != nil {
		return fmt.Errorf("could not determine executable path: %w", err)
	}
	execPath, err = filepath.Abs(execPath)
	if err != nil {
		return fmt.Errorf("could not resolve executable path: %w", err)
	}

	vars := daemonVars{
		ExecPath:    execPath,
		Port:        cfg.Port,
		Host:        cfg.Host,
		Region:      cfg.Region,
		Model:       cfg.Model,
		EmbedModel:  cfg.EmbedModel,
		BearerToken: cfg.BearerToken,
		BaseURL:     cfg.BaseURL,
	}

	unitPath := "/etc/systemd/system/bedrockify.service"

	if _, err := os.Stat("/etc/systemd/system"); os.IsNotExist(err) {
		return fmt.Errorf("/etc/systemd/system does not exist — is systemd available?")
	}

	f, err := os.Create(unitPath)
	if err != nil {
		return fmt.Errorf("failed to write %s (try running as root): %w", unitPath, err)
	}
	defer f.Close()

	tmpl, err := template.New("service").Parse(daemonTemplate)
	if err != nil {
		return fmt.Errorf("template parse: %w", err)
	}
	if err := tmpl.Execute(f, vars); err != nil {
		return fmt.Errorf("template execute: %w", err)
	}

	fmt.Printf("✓ Wrote %s\n", unitPath)
	fmt.Println("To enable and start the service:")
	fmt.Println("  sudo systemctl daemon-reload")
	fmt.Println("  sudo systemctl enable bedrockify")
	fmt.Println("  sudo systemctl start bedrockify")
	fmt.Println("  sudo systemctl status bedrockify")
	return nil
}
