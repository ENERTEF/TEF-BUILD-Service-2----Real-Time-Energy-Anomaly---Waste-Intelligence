"""Service 2: real-time energy anomaly and waste intelligence.

Modules follow the specification's pipeline, each independently importable:

  config     thresholds and model version, stored with every output
  occupancy  public holidays, school closures, occupancy windows
  sources    the data contract: what a deployment must provide (DataSource)
  ingest     15-minute meter series, names, €/kWh from bills, weather, PV, indoor temperature
  quality    data quality control: which meters are fit for detection, and meter-fault events
  baseline   per-building expected consumption, matched on day type and degree days
  detect     interval scoring, event typing, confidence, annualised waste and trend
  evaluate   walk-forward baseline accuracy and synthetic-event detection tests
  run        orchestration, progress log, versioned outputs, markdown report

Entry point:  waste_intel.run.run(data_source)
"""
