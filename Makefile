.PHONY: setup graph load calibrate run ui all clean

setup:      ; pip install -r requirements.txt
graph:      ; python -m agent.graph_build
load:       ; python -m agent.load_tigergraph --all
calibrate:  ; python -m agent.calibrate
run:        ; python -m agent.run_cases
run-tg:     ; python -m agent.run_cases --backend tigergraph --write-graph
ui:         ; python ui/build_dashboard.py
all: graph load calibrate run-tg ui
clean:      ; rm -rf data/graph cases/*.json ui/dashboard.html
