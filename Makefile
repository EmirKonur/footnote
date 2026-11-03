# Footnote: common tasks. `make help` lists them.
-include .env
export

API := cd api &&
WEB := cd web &&

.PHONY: help install dev api web seed fixtures lint typecheck test test-api test-web e2e \
        eval eval-gate eval-generation bench build check docker-up docker-down clean

help: ## Show this list
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-16s %s\n", $$1, $$2}'

install: ## Install API and web dependencies from the lockfiles
	$(API) uv sync --locked
	$(WEB) pnpm install --frozen-lockfile

dev: ## Run the API (4410) and web app (4411) together
	@$(MAKE) -j2 api web

api: ## Run the API with reload on :4410
	$(API) uv run footnote serve --reload

web: ## Run the web app in dev mode on :4411
	$(WEB) pnpm dev

seed: ## Ingest the bundled Python PEP corpus as a collection
	$(API) uv run footnote seed

fixtures: ## Run the API with fixture answers (no model calls) on :4410
	$(API) FOOTNOTE_FAKE_LLM=fixtures uv run footnote serve --reload

lint: ## Ruff and ESLint
	$(API) uv run ruff check src tests scripts ../evals/corpus && uv run ruff format --check src tests scripts ../evals/corpus
	$(WEB) pnpm lint

typecheck: ## Pyright (strict) and tsc
	$(API) uv run pyright
	$(WEB) pnpm typecheck

test-api: ## Backend tests
	$(API) uv run pytest

test-web: ## Frontend unit tests
	$(WEB) pnpm test

test: test-api test-web ## All unit and integration tests

e2e: ## Playwright smoke tests (builds the web app first)
	$(WEB) pnpm build && pnpm e2e

eval: ## Offline retrieval eval; writes evals/reports/retrieval.{md,json}
	$(API) uv run footnote eval retrieval

eval-gate: ## Retrieval eval in CI mode: fail on regressions against the committed report
	$(API) uv run footnote eval retrieval --out ../evals/reports/ci \
		--baseline ../evals/reports/retrieval.json --tolerance 0.03 \
		--fail-under hybrid.recall@5=0.8

eval-generation: ## LLM-judged answer eval (calls the Anthropic API, costs money)
	$(API) uv run footnote eval generation

bench: ## Time exact vector search at several collection sizes
	$(API) uv run python scripts/bench_vectors.py

build: ## Production build of the web app
	$(WEB) pnpm build

check: lint typecheck test build ## Everything CI runs, except e2e and the eval gate

docker-up: ## Build and start the stack with docker compose
	docker compose up --build

docker-down: ## Stop the stack
	docker compose down

clean: ## Remove build output and caches (keeps data and downloaded models)
	rm -rf web/.next web/test-results web/playwright-report web/e2e/.data api/.pytest_cache api/.ruff_cache
