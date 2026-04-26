package config

import (
	"fmt"
	"os"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

type Config struct {
	Server    ServerConfig     `yaml:"server"`
	Proxies   []ProxyConfig    `yaml:"proxies"`
	Providers []ProviderConfig `yaml:"providers"`
	Models    []ModelConfig    `yaml:"models"`
	Gateway   GatewayConfig    `yaml:"gateway"`
}

type ServerConfig struct {
	Host            string        `yaml:"host"`
	Port            int           `yaml:"port"`
	ReadTimeout     time.Duration `yaml:"read_timeout"`
	WriteTimeout    time.Duration `yaml:"write_timeout"`
	ShutdownTimeout time.Duration `yaml:"shutdown_timeout"`
}

type ProviderConfig struct {
	Name     string   `yaml:"name"`
	Type     string   `yaml:"type"` // auto-detected from base_url if empty
	BaseURL  string   `yaml:"base_url"`
	APIKeys  []string `yaml:"api_keys"`
	Proxy    string   `yaml:"proxy"`
	Models   []string `yaml:"models"` // auto-discovered if empty
	Weight   int      `yaml:"weight"`
	Disabled bool     `yaml:"disabled"`
}

// ModelConfig defines a virtual model entry with backends.
// Each backend is "provider-name/real-model-name".
type ModelConfig struct {
	Name     string   `yaml:"name"`
	Backends []string `yaml:"backends"`
}

type GatewayConfig struct {
	BalancerType    string        `yaml:"balancer_type"`
	RetryMax        int           `yaml:"retry_max"`
	RetryWaitMin    time.Duration `yaml:"retry_wait_min"`
	RetryWaitMax    time.Duration `yaml:"retry_wait_max"`
	GatewayRetryMax int           `yaml:"gateway_retry_max"`
	HealthCheck     bool          `yaml:"health_check"`
	HealthInterval  time.Duration `yaml:"health_interval"`
	HealthTimeout   time.Duration `yaml:"health_timeout"`
	RequestTimeout  time.Duration `yaml:"request_timeout"`
	APIKeys         []string      `yaml:"api_keys"`
}

type ProxyConfig struct {
	Name            string        `yaml:"name"`
	Type            string        `yaml:"type"`
	Address         string        `yaml:"address"`
	Username        string        `yaml:"username"`
	Password        string        `yaml:"password"`
	MaxIdleConns    int           `yaml:"max_idle_conns"`
	MaxConnsPerHost int           `yaml:"max_conns_per_host"`
	IdleConnTimeout time.Duration `yaml:"idle_conn_timeout"`
}

func DefaultConfig() *Config {
	return &Config{
		Server: ServerConfig{
			Host:            "127.0.0.1",
			Port:            8080,
			ReadTimeout:     30 * time.Second,
			WriteTimeout:    600 * time.Second,
			ShutdownTimeout: 10 * time.Second,
		},
		Gateway: GatewayConfig{
			BalancerType:    "round_robin",
			RetryMax:        5,
			RetryWaitMin:    500 * time.Millisecond,
			RetryWaitMax:    5 * time.Second,
			GatewayRetryMax: 3,
			HealthCheck:     true,
			HealthInterval:  30 * time.Second,
			HealthTimeout:   5 * time.Second,
			RequestTimeout:  300 * time.Second,
		},
	}
}

func Load(path string) (*Config, error) {
	cfg := DefaultConfig()

	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return cfg, nil
		}
		return nil, fmt.Errorf("reading config file: %w", err)
	}

	if err := yaml.Unmarshal(data, cfg); err != nil {
		return nil, fmt.Errorf("parsing config file: %w", err)
	}
	return cfg, nil
}

func (c *Config) Validate() error {
	if c.Server.Port < 1 || c.Server.Port > 65535 {
		return fmt.Errorf("invalid server port: %d", c.Server.Port)
	}
	if len(c.Providers) == 0 {
		return fmt.Errorf("no providers configured")
	}

	// Validate proxies
	proxyNames := make(map[string]*ProxyConfig)
	for i := range c.Proxies {
		p := &c.Proxies[i]
		if p.Name == "" {
			return fmt.Errorf("proxy %d: name is required", i)
		}
		if p.Type != "http" && p.Type != "socks5" {
			return fmt.Errorf("proxy %q: type must be \"http\" or \"socks5\"", p.Name)
		}
		if p.Address == "" {
			return fmt.Errorf("proxy %q: address is required", p.Name)
		}
		if _, dup := proxyNames[p.Name]; dup {
			return fmt.Errorf("duplicate proxy name: %q", p.Name)
		}
		proxyNames[p.Name] = p
	}

	// Validate providers and build name set
	providerNames := make(map[string]bool)
	for i, p := range c.Providers {
		if p.Name == "" {
			return fmt.Errorf("provider %d: name is required", i)
		}
		if p.BaseURL == "" {
			return fmt.Errorf("provider %d: base_url is required", i)
		}
		if len(p.APIKeys) == 0 {
			return fmt.Errorf("provider %d: at least one api_key is required", i)
		}
		if p.Proxy != "" {
			if _, found := proxyNames[p.Proxy]; !found {
				return fmt.Errorf("provider %q: references unknown proxy %q", p.Name, p.Proxy)
			}
		}
		providerNames[p.Name] = true
	}

	// Validate model entries
	modelNames := make(map[string]bool)
	for i, m := range c.Models {
		if m.Name == "" {
			return fmt.Errorf("model %d: name is required", i)
		}
		if modelNames[m.Name] {
			return fmt.Errorf("duplicate model name: %q", m.Name)
		}
		modelNames[m.Name] = true

		if len(m.Backends) == 0 {
			return fmt.Errorf("model %q: at least one backend is required", m.Name)
		}
		for _, b := range m.Backends {
			parts := strings.SplitN(b, "/", 2)
			if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
				return fmt.Errorf("model %q: backend %q must be in \"provider/model\" format", m.Name, b)
			}
			if !providerNames[parts[0]] {
				return fmt.Errorf("model %q: backend references unknown provider %q", m.Name, parts[0])
			}
		}
	}

	return nil
}

func (c *Config) ResolveProxy(name string) *ProxyConfig {
	for i := range c.Proxies {
		if c.Proxies[i].Name == name {
			return &c.Proxies[i]
		}
	}
	return nil
}

func (c *Config) TotalKeyCount() int {
	total := 0
	for _, p := range c.Providers {
		if !p.Disabled {
			total += len(p.APIKeys)
		}
	}
	return total
}
