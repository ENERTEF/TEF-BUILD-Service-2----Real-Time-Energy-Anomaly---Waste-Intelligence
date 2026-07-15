# Real-Time Energy Anomaly & Waste Intelligence Service

**Service 2 — Technical Description**
Municipality of Athens

---

## 1. Service Description

Service 2 equips the Municipality of Athens with a continuous, automated intelligence layer for detecting abnormal consumption behaviour and quantifying energy waste across its public building portfolio. Rather than waiting for anomalies to surface through quarterly bill reviews or operator complaints, the service identifies deviations from expected consumption as they emerge in the live meter feed, ranks them by financial impact and presents building operators with a prioritised action queue.

The objective is to make invisible waste visible and measurable, enabling targeted corrective action before losses accumulate. Over time the service builds an evolving waste register across the estate that supports both operational decision-making and strategic reporting on energy efficiency progress.

---

## 2. Operational Context

The service covers the full portfolio of smart-metered municipal buildings in Athens, encompassing a wide range of building types with distinct occupancy rhythms, operational schedules and consumption profiles. This diversity means a single detection threshold applied uniformly across the estate would produce an unacceptable volume of false positives for some buildings while missing genuine waste events in others. The service therefore builds and maintains a **dedicated baseline consumption profile for each building**, calibrated to its specific use category, area, characteristics and seasonal behaviour and updates these profiles continuously as new meter data arrives.

The most operationally significant waste categories in the municipal estate are:

- **Ghost loads** persisting through nights and weekends
- **Heating and cooling systems** running against closed or unoccupied buildings
- **Post-holiday thermal recovery** consuming disproportionate energy relative to the temperature deficit to be recovered
- **Gradual equipment degradation** manifesting as a slow upward drift in consumption that no single meter reading would flag as anomalous

Weather conditions are a primary confounding factor: a genuine increase in heating consumption during a cold snap must be distinguished from the same apparent signal caused by a faulty control system.

For buildings with installed PV systems, net consumption computed from gross meter readings and inverter-level generation data is used throughout, ensuring solar generation events are not misclassified as anomalous consumption reductions and that self-consumption patterns do not distort the building baseline.

---

## 3. Data Description

### 3.1 Core Datasets

The service uses the core datasets of the building portfolio, as specified in Service 1.

**Buildings**

| Variable | Name | Type | Unit | Description | Example |
|---|---|---|---|---|---|
| Building name | `name` | String | – | Name of the building | Town Hall |
| Address | `address` | String | – | Street and number | Athinas 63 |
| Latitude | `latitude` | String | – | Latitude (Google Maps) | 37.973 |
| Longitude | `longitude` | String | – | Longitude (Google Maps) | 23.706 |
| Category | `category` | String | – | Building category | Services |
| Total area | `total_area` | Float | m² | Total building area | 105.5 |
| Built year | `built_year` | Integer | – | Construction year | 1990 |
| Supply number | `supply_number` | String | – | Energy supply number(s) | 11111111111 |

**Historical Energy Bills**

| Variable | Name | Type | Unit | Description | Example |
|---|---|---|---|---|---|
| Bill number | `bill_number` | Integer | – | Number of the bill | 1111111111 |
| Issue date | `bill_issue_date` | Date | YYYY-MM-DD | Bill issue date | 2023-02-28 |
| Meter reading date | `meter_reading_date` | Date | YYYY-MM-DD | Date of meter reading | 2023-02-28 |
| Previous reading date | `previous_meter_reading_date` | Date | YYYY-MM-DD | Date of previous reading | 2023-02-28 |
| Consumption | `bill_kWh` | Integer | kWh | Total kWh for the billing period | 500 |
| Cost | `bill_total_euros` | Float | € | Total bill cost | 45.5 |
| Supply number | `supply_number` | String | – | Supply number(s) of the bill | 1111111111 |
| Bill type | `bill_type` | String | – | Type of bill | Settlement |

**Real-Time Energy Consumption**

| Variable | Name | Type | Unit | Description | Example |
|---|---|---|---|---|---|
| Building name | `building_name` | String | – | Name of the building | Town Hall |
| Supply number | `supply_number` | String | – | Energy supply number | 1111111111 |
| Timestamp | `timestamp_consumption` | Timestamp | YYYY-MM-DD HH:MM | Timestamp of consumption | 2023-02-28 00:15 |
| Consumption | `consumption` | Float | kWh | Building energy consumption | 1.2 |

### 3.2 Additional Monitored Streams

Beyond the core datasets, the service ingests and cross-correlates the following telemetry streams at 15-minute resolution:

| Stream | Resolution | Unit | Description |
|---|---|---|---|
| PV generation | 15 min | kWh | Inverter-level generation data for buildings with installed PV, used to compute net consumption |
| Diesel level | 15 min | litres (tank level) | Diesel tank level measurements for buildings with oil-fired heating, enabling detection of abnormal fuel draw-down, boiler inefficiency and potential leaks or losses |
| Water consumption | 15 min | litres | Building-level water metering, enabling detection of leaks, continuously running fixtures and abnormal consumption during unoccupied periods |
| Indoor temperature | 15 min | °C | Thermostat readings per building or zone, providing direct evidence of heating/cooling operation against unoccupied spaces and control system faults |

The multi-stream design extends waste detection beyond electricity alone. Diesel draw-down against thermostat readings and outdoor temperature exposes heating systems running inefficiently or against empty buildings; water flow during closed hours is a direct leak signature; and indoor temperature trajectories distinguish genuine thermal demand from control faults, replacing inference with direct measurement. Historical bills provide a long-horizon reference for validating baselines, detecting billing boundary artefacts and cross-checking cumulative waste estimates against actual expenditure.

---

## 4. Methodology and Technical Approach

The service constructs a rolling baseline for each building and each monitored stream using a lookback window of up to twelve months of historical data, adjusted for outdoor temperature through degree-day normalisation, stratified by day type and time-of-day interval and further segmented by occupancy status derived from building schedules and the Greek public holiday calendar. Each baseline represents the expected consumption distribution for a building under any given combination of contextual conditions, providing the statistical reference against which incoming readings are evaluated.

Anomaly detection operates at **15-minute resolution**, comparing each new interval against the corresponding baseline expectation and flagging deviations that exceed calibrated thresholds. Cross-stream corroboration strengthens classification: for example, a heating waste flag is supported by simultaneous evidence from electricity or diesel consumption, indoor temperature readings and occupancy status.

Detections are classified by type, covering:

- Ghost loads
- Unoccupied-period waste (electricity, heating fuel or water)
- Equipment fault signatures
- Water leaks and continuously running fixtures
- Abnormal diesel draw-down and boiler inefficiency
- Thermostat and control system faults (e.g. conditioning unoccupied spaces, setpoint violations)
- Post-holiday spikes
- Billing boundary artefacts
- Meter and sensor data quality issues, including resets and gaps

Each detection is assigned a **confidence score** reflecting the statistical strength of the deviation, the quality of the underlying data and the consistency of the signal across consecutive intervals and streams. Detections meeting a minimum confidence threshold are passed to the waste register with an estimated annual waste magnitude expressed in physical
