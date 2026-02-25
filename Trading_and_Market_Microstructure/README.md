# Trading and Market Microstructure

This folder contains execution-focused and signal-oriented research artifacts related to market microstructure, trading workflow design, and real-time analytics concepts.

It is intended to demonstrate:
- signal engineering intuition
- execution-aware thinking
- market data cleaning / structuring workflows
- dashboard-oriented monitoring concepts

---

## Files in This Folder

### `Signal_Engineering.ipynb`
Notebook for prototyping and evaluating simple trading signals using time-series features.

**Demonstrates:**
- feature engineering for trading signals
- rolling indicators / z-score style normalization
- signal labeling logic
- visualization of signal behavior over time

---

### `Execution_Models.ipynb`
Notebook exploring execution assumptions and transaction-cost-aware performance framing.

**Demonstrates:**
- slippage assumptions
- simple execution model comparisons
- cost-adjusted return intuition
- implementation-oriented research thinking

---

### `Data_Accumulation_Parsing_and_Cleaning.ipynb`
Notebook focused on collecting, parsing, and cleaning market-related datasets.

**Demonstrates:**
- data hygiene
- preprocessing workflow design
- schema consistency
- reproducible cleaning steps for modeling pipelines

---

### `Real_Time_Dashboards.py`
Python script for mock real-time dashboarding workflows.

**Demonstrates:**
- operational monitoring mindset
- visualization for live metrics
- trader/researcher workflow support

---

## What This Folder Signals

This section is designed to show that I can think beyond “just a model” and account for:
- how signals are built
- how execution affects realized outcomes
- how data quality impacts strategy logic
- how monitoring tools fit into a trading workflow

---

## Recommended Starting Point

1. `Signal_Engineering.ipynb`
2. `Execution_Models.ipynb`
3. `Real_Time_Dashboards.py`

---

## Next Planned Enhancements

- Add bid/ask spread proxies and explicit transaction cost modeling
- Add turnover estimates and signal decay diagnostics
- Add execution quality summary tables
- Add screenshots for portfolio site integration
