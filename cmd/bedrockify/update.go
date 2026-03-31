package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

const (
	updateRepo    = "inceptionstack/bedrockify"
	updateBinary  = "bedrockify"
	updateTimeout = 30 * time.Second
)

type githubRelease struct {
	TagName string `json:"tag_name"`
}

func runUpdate(currentVersion string) error {
	goos := runtime.GOOS
	goarch := runtime.GOARCH
	if goarch == "arm64" || goarch == "aarch64" {
		goarch = "arm64"
	} else if goarch == "amd64" || goarch == "x86_64" {
		goarch = "amd64"
	}

	fmt.Println("⬇️  Checking for updates...")
	latestTag, err := getLatestRelease()
	if err != nil {
		return fmt.Errorf("failed to check latest release: %w", err)
	}

	if latestTag == currentVersion {
		fmt.Printf("✅ Already on latest version (%s)\n", currentVersion)
		return nil
	}
	fmt.Printf("   Current: %s → Latest: %s\n", currentVersion, latestTag)

	execPath, err := os.Executable()
	if err != nil {
		return fmt.Errorf("could not determine executable path: %w", err)
	}
	execPath, err = filepath.EvalSymlinks(execPath)
	if err != nil {
		return fmt.Errorf("could not resolve executable path: %w", err)
	}

	url := fmt.Sprintf("https://github.com/%s/releases/latest/download/%s-%s-%s",
		updateRepo, updateBinary, goos, goarch)

	fmt.Printf("⬇️  Downloading %s-%s-%s...\n", updateBinary, goos, goarch)

	tmpFile, err := os.CreateTemp(filepath.Dir(execPath), ".bedrockify-update-*")
	if err != nil {
		tmpFile, err = os.CreateTemp("", ".bedrockify-update-*")
		if err != nil {
			return fmt.Errorf("failed to create temp file: %w", err)
		}
	}
	tmpPath := tmpFile.Name()
	defer os.Remove(tmpPath)

	client := &http.Client{Timeout: updateTimeout}
	resp, err := client.Get(url)
	if err != nil {
		tmpFile.Close()
		return fmt.Errorf("download failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		tmpFile.Close()
		return fmt.Errorf("download failed: HTTP %d", resp.StatusCode)
	}

	if _, err := io.Copy(tmpFile, resp.Body); err != nil {
		tmpFile.Close()
		return fmt.Errorf("download failed: %w", err)
	}
	tmpFile.Close()

	if err := os.Chmod(tmpPath, 0755); err != nil {
		return fmt.Errorf("chmod failed: %w", err)
	}

	needsSudo := false
	if err := os.Rename(tmpPath, execPath); err != nil {
		needsSudo = true
		cmd := exec.Command("sudo", "mv", tmpPath, execPath)
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			return fmt.Errorf("failed to replace binary (try running with sudo): %w", err)
		}
	}

	fmt.Printf("✅ Updated bedrockify to %s\n", latestTag)

	if isServiceActive("bedrockify") {
		fmt.Println("🔄 Restarting bedrockify daemon...")
		var cmd *exec.Cmd
		if needsSudo || os.Getuid() != 0 {
			cmd = exec.Command("sudo", "systemctl", "restart", "bedrockify")
		} else {
			cmd = exec.Command("systemctl", "restart", "bedrockify")
		}
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			log.Printf("⚠️  Failed to restart daemon: %v (restart manually: sudo systemctl restart bedrockify)", err)
		} else {
			fmt.Println("✅ Daemon restarted")
		}
	}

	return nil
}

func getLatestRelease() (string, error) {
	url := fmt.Sprintf("https://api.github.com/repos/%s/releases/latest", updateRepo)
	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("GitHub API returned %d", resp.StatusCode)
	}

	var release githubRelease
	if err := json.NewDecoder(resp.Body).Decode(&release); err != nil {
		return "", err
	}
	return release.TagName, nil
}

func isServiceActive(name string) bool {
	cmd := exec.Command("systemctl", "is-active", name)
	out, err := cmd.Output()
	if err != nil {
		return false
	}
	return strings.TrimSpace(string(out)) == "active"
}
