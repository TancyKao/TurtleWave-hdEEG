# Reference: EEG Review GUI

Complete technical reference for the TurtleWave EEG Event Review GUI.

## Overview

The EEG Review GUI ([`frontend.eeg_review_gui`](../../frontend/eeg_review_gui.py)) is a PyQt5-based application for reviewing and validating automatically detected EEG events (spindles, slow waves, etc.).

**Module:** `frontend.eeg_review_gui`

**Main Class:** [`EventReviewGUI`](../../frontend/eeg_review_gui.py:910)

**Launch Command:**

```bash
python -m frontend.eeg_review_gui
```

## GUI Layout

The interface consists of three main panels:

```
┌─────────────┬──────────────────────────────┬─────────────────┐
│   Channel   │        Timeline              │   Navigation    │
│   Selector  │        Event List            │   & Review      │
│             │        Waveform Plot         │   Controls      │
│   (Left)    │        (Middle)              │   (Right)       │
└─────────────┴──────────────────────────────┴─────────────────┘
```

## Main Components

### EventReviewGUI

Main application window.

**Class:** [`EventReviewGUI(QMainWindow)`](../../frontend/eeg_review_gui.py:910)

**Initialization:**

```python
gui = EventReviewGUI()
gui.show()
```

**Key Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `db` | `EventDatabase` | Database connection handler |
| `eeg_data` | `LargeDataset` | EEG data loader |
| `annotations` | `CustomAnnotations` | Sleep stage annotations |
| `current_events` | `pd.DataFrame` | Filtered event list |
| `current_event_index` | `int` | Currently displayed event index |
| `reviewer_name` | `str` | Name of current reviewer |
| `selected_channels` | `list[str]` | Channels to display |
| `selected_event_types` | `list[str]` | Event types to show |

**Key Methods:**

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `open_database()` | `db_path: str` | `None` | Load event database |
| `open_eeg_file()` | `eeg_path: str` | `None` | Load EEG data file |
| `open_annotation_file()` | `annot_path: str` | `None` | Load sleep annotations |
| `review_event()` | `decision: str` | `None` | Accept/reject current event |
| `next_event()` | - | `None` | Navigate to next event |
| `previous_event()` | - | `None` | Navigate to previous event |
| `jump_to_event()` | `index: int` | `None` | Jump to specific event |
| `apply_filters()` | - | `None` | Apply all active filters |
| `export_results()` | `output_path: str` | `None` | Export reviewed events |

### EventDatabase

Database handler for event storage and retrieval.

**Class:** [`EventDatabase`](../../frontend/eeg_review_gui.py:42)

**Initialization:**

```python
db = EventDatabase(db_path="events.db")
```

**Methods:**

#### `get_events()`

Retrieve events with optional filtering.

**Signature:**

```python
def get_events(
    self,
    event_type: str = None,
    channels: list[str] = None,
    stages: list[str] = None,
    reviewed_only: bool = False,
    unreviewed_only: bool = False,
    confidence_threshold: float = 0.0,
    methods: list[str] = None,
    freq_bands: list[tuple] = None
) -> pd.DataFrame
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `event_type` | `str` | `None` | Filter by event type (e.g., "spindle") |
| `channels` | `list[str]` | `None` | Filter by channel names |
| `stages` | `list[str]` | `None` | Filter by sleep stages |
| `reviewed_only` | `bool` | `False` | Show only reviewed events |
| `unreviewed_only` | `bool` | `False` | Show only unreviewed events |
| `confidence_threshold` | `float` | `0.0` | Minimum confidence score (0-1) |
| `methods` | `list[str]` | `None` | Filter by detection method |
| `freq_bands` | `list[tuple]` | `None` | Filter by frequency band |

**Returns:** `pd.DataFrame` with columns:

- `uuid`: Unique event identifier
- `event_type`: Type of event (spindle, slow_wave, etc.)
- `start_time`: Event start time (seconds)
- `end_time`: Event end time (seconds)
- `duration`: Event duration (seconds)
- `channel`: Channel name
- `stage`: Sleep stage
- `confidence`: Detection confidence (0-1)
- `amplitude`: Peak amplitude (µV)
- `frequency`: Dominant frequency (Hz)
- `method`: Detection method
- `review_decision`: "accept", "reject", or NULL
- `reviewer`: Reviewer name
- `review_timestamp`: ISO 8601 timestamp

**Example:**

```python
# Get all unreviewed spindles from N2 sleep
events = db.get_events(
    event_type="spindle",
    stages=["N2"],
    unreviewed_only=True,
    confidence_threshold=0.7
)
```

#### `add_review()`

Add review decision for an event.

**Signature:**

```python
def add_review(
    self,
    uuid: str,
    decision: str,
    reviewer: str = "",
    comments: str = ""
) -> None
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `uuid` | `str` | Event UUID |
| `decision` | `str` | "accept" or "reject" |
| `reviewer` | `str` | Reviewer name |
| `comments` | `str` | Optional comments |

**Example:**

```python
db.add_review(
    uuid="abc123",
    decision="accept",
    reviewer="Reviewer1",
    comments="Clear spindle"
)
```

#### `get_review_stats()`

Get review statistics.

**Signature:**

```python
def get_review_stats(self) -> dict
```

**Returns:** Dictionary with keys:

- `total`: Total events in database
- `reviewed`: Number of reviewed events
- `accepted`: Number of accepted events
- `rejected`: Number of rejected events
- `percent_complete`: Percentage reviewed (0-100)

**Example:**

```python
stats = db.get_review_stats()
print(f"Progress: {stats['percent_complete']:.1f}%")
print(f"Accepted: {stats['accepted']}, Rejected: {stats['rejected']}")
```

#### `export_reviewed_events()`

Export reviewed events to CSV.

**Signature:**

```python
def export_reviewed_events(self, output_path: str) -> None
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `output_path` | `str` | Path to output CSV file |

**Example:**

```python
db.export_reviewed_events("reviewed_events.csv")
```

### EventTableModel

Virtualized table model for efficient display of large event lists.

**Class:** [`EventTableModel(QAbstractTableModel)`](../../frontend/eeg_review_gui.py:270)

**Features:**

- Virtualized rendering for 100,000+ events
- Sortable columns
- Custom formatting for timestamps and numeric values
- Color-coded review status

**Methods:**

| Method | Description |
|--------|-------------|
| `set_events(events_df)` | Update displayed events |
| `rowCount()` | Number of events |
| `columnCount()` | Number of columns |
| `data(index, role)` | Get cell data |
| `headerData(section, orientation, role)` | Get header labels |

### TimelinePlot

Interactive timeline showing sleep hypnogram.

**Class:** [`TimelinePlot(PlotWidget)`](../../frontend/eeg_review_gui.py:393)

**Features:**

- Sleep stage visualization (color-coded)
- Clickable timeline for navigation
- Current event indicator
- Time axis with HH:MM:SS formatting

**Methods:**

| Method | Parameters | Description |
|--------|------------|-------------|
| `plot_timeline()` | `events_df, current_index, annotations` | Draw timeline |
| `on_click()` | `event` | Handle mouse click |

**Sleep Stage Colors:**

| Stage | Color | RGB |
|-------|-------|-----|
| Wake | Yellow | (255, 255, 0) |
| N1 | Light Blue | (173, 216, 230) |
| N2 | Blue | (0, 0, 255) |
| N3 | Dark Blue | (0, 0, 139) |
| REM | Green | (0, 255, 0) |

### WaveformPlot

EEG waveform display with event highlighting.

**Class:** [`WaveformPlot(PlotWidget)`](../../frontend/eeg_review_gui.py:594)

**Features:**

- Multi-channel display with automatic spacing
- Event highlighting (shaded region)
- Configurable time window (10-60 seconds)
- Optional bandpass filtering
- Scale bar (50 µV)
- Channel labels

**Methods:**

| Method | Parameters | Description |
|--------|------------|-------------|
| `plot_event()` | `event_row, waveform_data, channels` | Plot event waveforms |
| `set_window_duration()` | `duration` | Set time window (seconds) |
| `toggle_filter()` | `enabled` | Enable/disable filtering |
| `apply_filter()` | `data` | Apply bandpass filter |

**Filter Settings:**

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Low frequency | 0.5 Hz | 0.1-30 Hz | High-pass cutoff |
| High frequency | 30 Hz | 1-100 Hz | Low-pass cutoff |
| Filter order | 2 | - | Butterworth filter order |

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `A` | Accept current event |
| `R` | Reject current event |
| `→` | Next event |
| `←` | Previous event |
| `Space` | Next unreviewed event |
| `Ctrl+O` | Open database |
| `Ctrl+E` | Open EEG file |
| `Ctrl+S` | Export results |
| `Ctrl+Q` | Quit application |

## Menu Bar

### File Menu

| Item | Shortcut | Action |
|------|----------|--------|
| Open Database... | `Ctrl+O` | Load event database |
| Open EEG File... | `Ctrl+E` | Load EEG data |
| Open Annotation File... | - | Load sleep annotations |
| Exit | `Ctrl+Q` | Close application |

### Review Menu

| Item | Shortcut | Action |
|------|----------|--------|
| Accept Event | `A` | Accept current event |
| Reject Event | `R` | Reject current event |

### Export Menu

| Item | Shortcut | Action |
|------|----------|--------|
| Export Reviewed Events... | `Ctrl+S` | Export to CSV |

## Toolbar

| Button | Icon | Action |
|--------|------|--------|
| ◀ Prev | - | Previous event |
| Next ▶ | - | Next event |
| ✓ Accept | - | Accept event |
| ✗ Reject | - | Reject event |

## Filter Panel

### Event Type Filter

**Widget:** Checkbox group

**Options:** Dynamically loaded from database (e.g., spindle, slow_wave, ripple)

**Behavior:** Shows events matching ANY selected type (OR logic)

### Sleep Stage Filter

**Widget:** Checkbox group

**Options:**

- Wake
- N1
- N2
- N3
- REM

**Behavior:** Shows events from ANY selected stage (OR logic)

### Channel Filter

**Widget:** Text input (comma-separated)

**Format:** `E112, E118, Cz` or `Fp1, Fp2, F3, F4`

**Behavior:** Shows events from ANY listed channel (OR logic)

### Confidence Threshold

**Widget:** Slider (0.0 - 1.0)

**Default:** 0.0 (show all)

**Behavior:** Shows events with confidence ≥ threshold

### Review Status Filter

**Widget:** Checkbox group

**Options:**

- Reviewed Only
- Unreviewed Only

**Behavior:** Mutually exclusive filters

### Method Filter

**Widget:** Checkbox group

**Options:** Dynamically loaded from database (e.g., wavelet, bandpass, hilbert)

**Behavior:** Shows events from ANY selected method (OR logic)

### Frequency Band Filter

**Widget:** Checkbox group

**Options:** Dynamically loaded from database (e.g., 11-13 Hz, 13-16 Hz)

**Behavior:** Shows events from ANY selected band (OR logic)

## Display Settings

### Window Duration

**Widget:** Slider (10-60 seconds)

**Default:** 30 seconds

**Description:** Time window shown around event (centered on event midpoint)

### Filter Settings

**Enable Filter:** Checkbox (default: enabled)

**Low Frequency:** Slider (0.1-30 Hz, default: 0.5 Hz)

**High Frequency:** Slider (1-100 Hz, default: 30 Hz)

**Description:** Bandpass filter applied to displayed waveforms (does not affect raw data)

## Channel Selector

**Widget:** Tree widget with checkboxes

**Features:**

- Hierarchical channel organization
- Multi-select with checkboxes
- Quick select buttons (All, None, Default)
- Search/filter capability

**Default Channels:** E112, E118, Cz

## Review Statistics Panel

**Location:** Right panel, bottom

**Displays:**

| Metric | Description |
|--------|-------------|
| Total Events | Events matching current filters |
| Reviewed | Events with review decisions |
| Accepted | Events marked as accepted |
| Rejected | Events marked as rejected |
| % Complete | Percentage of events reviewed |

**Updates:** Real-time after each review decision

## Data Formats

### Input Files

**Event Database:**

- Format: SQLite database (`.db`)
- Required tables: `events`, `review_decisions`
- Created by: Event detection scripts

**EEG Data:**

- Formats: EEGLAB (`.set`, `.fdt`), EDF (`.edf`)
- Loader: [`LargeDataset`](../../turtlewave_hdEEG/dataset.py)

**Annotations:**

- Format: XML (`.xml`)
- Schema: Sleep stage annotations with start/end times
- Loader: [`CustomAnnotations`](../../turtlewave_hdEEG/annotation.py)

### Output Files

**Reviewed Events CSV:**

Columns:

- `uuid`: Event identifier
- `event_type`: Event type
- `start_time`: Start time (seconds)
- `end_time`: End time (seconds)
- `duration`: Duration (seconds)
- `channel`: Channel name
- `stage`: Sleep stage
- `confidence`: Detection confidence
- `amplitude`: Peak amplitude (µV)
- `frequency`: Dominant frequency (Hz)
- `method`: Detection method
- `review_decision`: "accept" or "reject"
- `reviewer`: Reviewer name
- `review_timestamp`: ISO 8601 timestamp
- `comments`: Optional comments

## Performance Characteristics

### Scalability

| Dataset Size | Load Time | Navigation Speed | Memory Usage |
|--------------|-----------|------------------|--------------|
| 1,000 events | < 1s | Instant | ~50 MB |
| 10,000 events | < 2s | Instant | ~100 MB |
| 100,000 events | < 5s | Instant | ~500 MB |
| 1,000,000 events | < 30s | < 100ms | ~2 GB |

### Optimization Features

- **Virtualized table rendering** - Only visible rows rendered
- **Background waveform loading** - Non-blocking data loading
- **Waveform caching** - Recently viewed events cached
- **Debounced filtering** - Filter updates batched
- **Lazy timeline rendering** - Timeline drawn once, not per event

## Database Schema

### events Table

```sql
CREATE TABLE events (
    uuid TEXT PRIMARY KEY,
    event_type TEXT,
    start_time REAL,
    end_time REAL,
    duration REAL,
    channel TEXT,
    stage TEXT,
    confidence REAL,
    amplitude REAL,
    frequency REAL,
    method TEXT,
    freq_band_lower REAL,
    freq_band_upper REAL
);
```

### review_decisions Table

```sql
CREATE TABLE review_decisions (
    uuid TEXT PRIMARY KEY,
    review_decision TEXT,
    reviewer TEXT,
    review_timestamp TEXT,
    comments TEXT,
    FOREIGN KEY (uuid) REFERENCES events(uuid)
);
```

### Indexes

```sql
CREATE INDEX idx_event_type ON events(event_type);
CREATE INDEX idx_channel ON events(channel);
CREATE INDEX idx_stage ON events(stage);
CREATE INDEX idx_confidence ON events(confidence);
CREATE INDEX idx_start_time ON events(start_time);
```

## Error Handling

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `FileNotFoundError` | Database/EEG file not found | Check file path |
| `sqlite3.DatabaseError` | Corrupted database | Regenerate database |
| `ValueError: No events found` | Empty database or filters too restrictive | Clear filters |
| `MemoryError` | Insufficient RAM | Reduce window duration or filter events |
| `ImportError: No module named 'mne'` | Missing dependency | `pip install mne` |

### Logging

**Log Level:** INFO

**Log Location:** Console output

**Log Format:** `[TIMESTAMP] LEVEL: Message`

**Example:**

```
[2026-03-01 09:00:00] INFO: Database loaded: 15234 events
[2026-03-01 09:00:05] INFO: EEG file loaded: 256 channels, 500 Hz
[2026-03-01 09:00:10] INFO: Review decision saved: accept
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TURTLEWAVE_CACHE_SIZE` | 100 | Number of waveforms to cache |
| `TURTLEWAVE_THREAD_COUNT` | 4 | Background loader threads |

### Config File

Not currently supported. Configuration is hardcoded in the application.

## Dependencies

**Required:**

- Python ≥ 3.7
- PyQt5 ≥ 5.12
- pyqtgraph ≥ 0.12
- pandas ≥ 1.0
- numpy ≥ 1.18
- scipy ≥ 1.4
- sqlite3 (standard library)

**Optional:**

- mne ≥ 0.20 (for EEG file loading)

## See Also

- [Tutorial: Your First EEG Event Review Session](../tutorials/eeg-review-gui-tutorial.md)
- [How-to Guide: Review EEG Events](../how-to/review-eeg-events.md)
- [Explanation: Review GUI Architecture](../explanation/eeg-review-gui-architecture.md)
- [API Reference: EventProcessor](./api/eventprocessor.md)
