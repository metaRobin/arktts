.PHONY: all cli server run-cli run-server clean

CMD_DIR := cmd
CLI_BIN := $(CMD_DIR)/arktts-cli
SERVER_BIN := $(CMD_DIR)/arktts-server

all: cli server

cli:
	go build -o $(CLI_BIN) ./cmd/arktts

server:
	go build -o $(SERVER_BIN) ./cmd/server

run-cli: cli
	cd $(CMD_DIR) && ./arktts-cli $(ARGS)

run-server: server
	cd $(CMD_DIR) && ./arktts-server

clean:
	rm -f $(CLI_BIN) $(SERVER_BIN)
