# Makefile for vectordb-go.
#
# The most important target is `proto`, which regenerates the gRPC Go code from
# api/proto/vectordb.proto. The generated code is checked in, so you only need
# this after editing the .proto file — everyday builds don't require protoc.

GO      ?= go
GOBIN   := $(shell $(GO) env GOPATH)/bin
PROTOC  ?= protoc

.PHONY: all build test vet fmt proto tools bench run clean

all: fmt vet test build

## build: compile the server and benchmark binaries into ./bin
build:
	$(GO) build -o bin/vectordb-server ./cmd/vectordb-server
	$(GO) build -o bin/vectordb-bench  ./cmd/vectordb-bench

## test: run all unit and integration tests
test:
	$(GO) test ./...

## vet: run go vet static checks
vet:
	$(GO) vet ./...

## fmt: format all Go code
fmt:
	$(GO) fmt ./...

## tools: install the protoc Go plugins (needs Go + network)
tools:
	$(GO) install google.golang.org/protobuf/cmd/protoc-gen-go@latest
	$(GO) install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

## proto: regenerate gRPC code from the .proto (needs protoc + `make tools`)
proto:
	PATH="$(PATH):$(GOBIN)" $(PROTOC) \
		--proto_path=api/proto \
		--go_out=api/gen --go_opt=paths=source_relative \
		--go-grpc_out=api/gen --go-grpc_opt=paths=source_relative \
		api/proto/vectordb.proto

## bench: run the benchmark tool (recall@k, QPS, build time, memory)
bench:
	$(GO) run ./cmd/vectordb-bench

## run: start the server with defaults (data in ./data)
run:
	$(GO) run ./cmd/vectordb-server

clean:
	rm -rf bin data
