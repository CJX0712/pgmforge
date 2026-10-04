# PgmForge 常用任务
# 作者：晨星

PY ?= python
LOCK := requirements.lock.txt

.PHONY: help install lock demo test cov lint fmt all docker clean

help:
	@echo "PgmForge · 概率图模型推断系统（作者 晨星）"
	@echo ""
	@echo "  make install   安装锁定依赖"
	@echo "  make demo      跑端到端基准并落盘 benchmark.json"
	@echo "  make test      跑测试套件"
	@echo "  make cov       跑测试 + 覆盖率报告（需 ≥80%）"
	@echo "  make lint      ruff 静态检查"
	@echo "  make fmt       ruff 格式化"
	@echo "  make all       lint + cov + demo（交付前全跑一遍）"
	@echo "  make docker    构建容器镜像"

install:
	$(PY) -m pip install -r $(LOCK)

demo:
	$(PY) cli.py demo

test:
	$(PY) -m pytest tests/ -q

cov:
	$(PY) -m pytest tests/ -q --cov=pgmforge --cov-report=term-missing --cov-fail-under=80

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

fmt:
	$(PY) -m ruff format .

all: lint cov demo

docker:
	docker build -t pgmforge:0.1.0 .

clean:
	@rm -rf .pytest_cache .ruff_cache html.xml .coverage
	@find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	@echo cleaned
