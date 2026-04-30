# Explanation: EEG Review GUI Architecture

This document explains the design principles, architecture, and technical decisions behind the TurtleWave EEG Review GUI.

## Purpose and Design Philosophy

### Why a Dedicated Review GUI?

Automated EEG event detection algorithms are powerful but imperfect. They produce false positives (detecting events that aren't real) and false negatives (missing real events). Manual review by trained experts remains the gold standard for validating detected events.

The EEG Review GUI was designed to make this manual review process:

- **Efficient** - Review hundreds or thousands of events quickly
- **Consistent** - Maintain uniform criteria across reviewers and sessions
- **Scalable** - Handle datasets with 100,000+ events without performance degradation
- **Reproducible** - Track all review decisions with timestamps and reviewer information

### Design Principles

**1. Keyboard-First Interaction**

The GUI prioritizes keyboard shortcuts over mouse clicks. Experienced reviewers can process events at a steady rhythm using only the keyboard (A for accept, R for reject, arrow keys for navigation). This reduces fatigue and increases throughput.

**2. Information Density Without Clutter**

The three-panel layout maximizes information density while maintaining clarity:

- **Left panel** - Channel selection (used occasionally)
- **Middle panel** - Primary focus (timeline, event list, waveforms)
- **Right panel** - Controls and filters (used periodically)

This layout minimizes eye movement and keeps the most important information (the waveform) in the center of the screen.

**3. Progressive Disclosure**

Advanced features are hidden until needed. The default view shows essential information, while filters, statistics, and settings are available in collapsible panels. This prevents overwhelming new users while providing power users with full control.

**4. Performance Over Features**

The GUI is optimized for speed. Virtualized rendering, background loading, and caching ensure that navigation feels instant even with massive datasets. Features that would compromise performance (like real-time event markers on the timeline) are deliberately omitted.

## Architecture Overview

### Component Hierarchy

```
EventReviewGUI (QMainWindow)
├── Menu Bar
│   ├── File Menu (Open, Export, Exit)
│   ├── Review Menu (Accept, Reject)
│   └── Export Menu (Export Results)
├── Toolbar (Navigation & Review buttons)
├── Left Panel (Channel Selector)
│   └── QTreeWidget (Channel list)
├── Middle Panel (Main Display)
│   ├── TimelinePlot (Sleep hypnogram)
│   ├── EventTableModel + QTableView (Event list)
│   └── WaveformPlot (EEG traces)
├── Right Panel (Controls)
│   ├── Navigation Controls
│   ├── Event Type Filter
│   ├── Sleep Stage Filter
│   ├── Channel Filter
│   ├── Confidence Threshold
│   ├── Review Status Filter
│   ├── Method Filter
│   ├── Frequency Band Filter
│   ├── Display Settings
│   └── Review Statistics
└── Status Bar
```

### Data Flow

```
┌─────────────┐
│   SQLite    │
│  Database   │
└──────┬──────┘
       │
       ↓
┌─────────────┐      ┌──────────────┐
│ Event       │      │  EEG Data    │
│ Database    │      │  (LargeDataset)│
└──────┬──────┘      └──────┬───────┘
       │                    │
       ↓                    ↓
┌─────────────────────────────────┐
│      EventReviewGUI             │
│  ┌──────────────────────────┐   │
│  │  Filter & Sort Events    │   │
│  └────────┬─────────────────┘   │
│           ↓                     │
│  ┌──────────────────────────┐   │
│  │  Display Current Event   │   │
│  └────────┬─────────────────┘   │
│           ↓                     │
│  ┌──────────────────────────┐   │
│  │  User Review Decision    │   │
│  └────────┬─────────────────┘   │
│           ↓                     │
│  ┌──────────────────────────┐   │
│  │  Save to Database        │   │
│  └──────────────────────────┘   │
└─────────────────────────────────┘
       │
       ↓
┌─────────────┐
│  Export CSV │
└─────────────┘
```

## Key Components Explained

### EventDatabase: The Data Layer

The [`EventDatabase`](../../frontend/eeg_review_gui.py:42) class abstracts all database operations. It provides a clean interface for:

- **Querying events** with complex filters
- **Saving review decisions** atomically
- **Computing statistics** efficiently
- **Exporting results** in standard formats

**Why SQLite?**

SQLite was chosen because:

- **Zero configuration** - No server setup required
- **Portable** - Single file contains all data
- **Fast** - Optimized for read-heavy workloads
- **Reliable** - ACID transactions ensure data integrity
- **Queryable** - Standard SQL for complex filters

The database uses indexes on frequently filtered columns (event_type, channel, stage, confidence) to maintain fast query performance even with millions of events.

### EventTableModel: Virtualized Rendering

The [`EventTableModel`](../../frontend/eeg_review_gui.py:270) implements Qt's Model-View architecture with virtualization. Instead of creating widgets for every event, it only renders visible rows.

**How Virtualization Works:**

1. The model stores the full DataFrame in memory
2. The view requests data only for visible rows (typically 20-50)
3. As the user scrolls, the view requests new rows
4. Old rows are discarded, keeping memory usage constant

This allows the GUI to display 100,000+ events with the same memory footprint as 100 events.

**Trade-offs:**

- **Pro:** Constant memory usage, instant scrolling
- **Con:** Sorting requires re-sorting the entire DataFrame (still fast with pandas)

### TimelinePlot: Context Visualization

The [`TimelinePlot`](../../frontend/eeg_review_gui.py:393) provides temporal context by showing the sleep hypnogram. This helps reviewers understand:

- **Sleep architecture** - Distribution of sleep stages
- **Event timing** - When events occur relative to sleep stages
- **Recording quality** - Gaps or artifacts in the recording

**Design Decision: No Event Markers**

Early versions plotted individual events on the timeline. This was removed because:

- **Performance** - Plotting 10,000+ markers caused lag
- **Visual clutter** - Dense events created an unreadable mess
- **Limited utility** - Reviewers focus on one event at a time

Instead, the timeline shows only the current event position (a vertical line), which updates instantly.

### WaveformPlot: The Core Display

The [`WaveformPlot`](../../frontend/eeg_review_gui.py:594) is where reviewers spend most of their time. It displays multi-channel EEG traces with the detected event highlighted.

**Key Features:**

**1. Configurable Time Window**

The window duration (10-60 seconds) is adjustable because different event types need different context:

- **Spindles** (0.5-2s duration) - Short window (10-15s) focuses attention
- **Slow waves** (0.5-1.5s duration) - Medium window (20-30s) shows wave context
- **Long events** - Longer window (40-60s) captures full event

**2. Optional Filtering**

The bandpass filter is optional because:

- **Spindles** benefit from 11-16 Hz filtering (removes slow drift)
- **Slow waves** benefit from 0.5-4 Hz filtering (removes high-frequency noise)
- **Artifacts** are easier to spot in unfiltered data

The filter is applied only for display - raw data remains unchanged.

**3. Automatic Channel Spacing**

Channels are automatically spaced to prevent overlap. The spacing algorithm:

1. Calculates the peak-to-peak amplitude of each channel
2. Adds 20% padding
3. Stacks channels vertically with consistent spacing

This ensures clean visualization regardless of signal amplitude.

**4. Event Highlighting**

The detected event is highlighted with a semi-transparent shaded region. This draws attention to the event without obscuring the waveform.

### Background Loading: Responsiveness

The [`WaveformBackgroundLoader`](../../frontend/waveform_loader.py) loads EEG data in background threads to keep the GUI responsive.

**How It Works:**

1. User navigates to a new event
2. GUI immediately displays the event metadata (from database)
3. Background thread loads waveform data from disk
4. When loading completes, the waveform plot updates
5. Recently loaded waveforms are cached for instant re-display

**Why Threading?**

Loading EEG data from disk can take 100-500ms for large files. Without threading, the GUI would freeze during navigation. Threading allows:

- **Immediate feedback** - Event metadata appears instantly
- **Smooth navigation** - Users can continue navigating while data loads
- **Caching** - Frequently accessed events load from memory

**Thread Safety:**

The loader uses Qt's signal-slot mechanism for thread-safe communication. Background threads emit signals when data is ready, and the main thread updates the GUI.

## Performance Optimizations

### 1. Debounced Filtering

When users adjust filters (e.g., typing channel names), the GUI doesn't re-query the database on every keystroke. Instead, it waits 300ms after the last change before applying filters. This reduces unnecessary database queries.

### 2. Lazy Timeline Rendering

The timeline is drawn once when data is loaded, not on every event navigation. Only the current event indicator (a vertical line) is updated, which is much faster than redrawing the entire plot.

### 3. Waveform Caching

The 100 most recently viewed events are cached in memory. When users navigate back to a previously viewed event, the waveform appears instantly without disk I/O.

### 4. Indexed Database Queries

All frequently filtered columns have database indexes:

```sql
CREATE INDEX idx_event_type ON events(event_type);
CREATE INDEX idx_channel ON events(channel);
CREATE INDEX idx_stage ON events(stage);
CREATE INDEX idx_confidence ON events(confidence);
```

This makes filtered queries fast even with millions of events.

### 5. Pandas for Data Manipulation

The GUI uses pandas DataFrames for event data because pandas is highly optimized for:

- **Filtering** - Boolean indexing is fast
- **Sorting** - Efficient sorting algorithms
- **Aggregation** - Fast statistics computation

## Design Trade-offs

### Trade-off 1: Memory vs. Speed

**Decision:** Load all filtered events into memory

**Rationale:** Modern computers have sufficient RAM (8-16 GB) to hold even large event lists (1 million events ≈ 500 MB). Loading all events enables instant sorting and filtering without database round-trips.

**Alternative:** Query database on-demand (slower but lower memory)

### Trade-off 2: Flexibility vs. Simplicity

**Decision:** Fixed three-panel layout

**Rationale:** A consistent layout reduces cognitive load and allows muscle memory to develop. Users always know where to find controls.

**Alternative:** Customizable layout (more flexible but more complex)

### Trade-off 3: Features vs. Performance

**Decision:** Omit real-time event markers on timeline

**Rationale:** Plotting thousands of markers causes noticeable lag. The timeline's primary purpose (showing sleep architecture) is achieved without markers.

**Alternative:** Plot all events (more informative but slower)

### Trade-off 4: Validation vs. Speed

**Decision:** Minimal input validation

**Rationale:** The GUI is designed for expert users who understand the data. Extensive validation would slow down the workflow.

**Alternative:** Strict validation (safer but slower)

## Why PyQt5 and pyqtgraph?

### PyQt5

**Advantages:**

- **Native performance** - C++ backend for fast rendering
- **Cross-platform** - Works on Windows, macOS, Linux
- **Mature ecosystem** - Extensive documentation and community
- **Professional appearance** - Native look and feel

**Disadvantages:**

- **Licensing** - GPL or commercial license required
- **Learning curve** - Complex API for beginners

**Alternatives considered:**

- **Tkinter** - Too slow for large datasets
- **wxPython** - Less mature plotting libraries
- **Web-based (Dash/Streamlit)** - Network latency, harder to deploy

### pyqtgraph

**Advantages:**

- **Fast** - GPU-accelerated rendering
- **Scientific focus** - Designed for data visualization
- **PyQt integration** - Seamless integration with PyQt5
- **Interactive** - Built-in zoom, pan, click handling

**Disadvantages:**

- **Limited styling** - Less customizable than matplotlib
- **Smaller community** - Fewer examples and tutorials

**Alternatives considered:**

- **matplotlib** - Too slow for real-time updates
- **plotly** - Requires web browser, harder to embed
- **vispy** - More complex API, less mature

## Extensibility

The GUI is designed for extension:

### Adding New Filters

New filters can be added by:

1. Adding UI widgets in [`create_right_panel()`](../../frontend/eeg_review_gui.py:1200)
2. Updating [`apply_filters()`](../../frontend/eeg_review_gui.py:1800) to include new criteria
3. Modifying [`EventDatabase.get_events()`](../../frontend/eeg_review_gui.py:119) if database queries are needed

### Adding New Event Types

The GUI automatically detects event types from the database. No code changes are needed to support new event types (e.g., K-complexes, artifacts).

### Custom Visualizations

New plot types can be added by:

1. Creating a new class inheriting from `pg.PlotWidget`
2. Adding the plot to the middle panel layout
3. Connecting it to the event navigation signals

## Common Misconceptions

### "Why not use a web interface?"

Web interfaces (Dash, Streamlit, Jupyter) are great for exploration but have limitations for production review:

- **Latency** - Network round-trips slow down navigation
- **Deployment** - Requires server setup and maintenance
- **Offline use** - Requires internet connection
- **Performance** - JavaScript rendering is slower than native code

Desktop applications provide better performance and user experience for intensive review tasks.

### "Why not integrate with existing EEG software?"

Existing EEG software (EEGLAB, MNE-Python, FieldTrip) is designed for analysis, not high-throughput review. They lack:

- **Batch review workflows** - No keyboard-driven rapid review
- **Review tracking** - No built-in decision logging
- **Performance optimization** - Not designed for 100,000+ events
- **Filtering flexibility** - Limited multi-criteria filtering

The Review GUI is purpose-built for the specific task of validating detected events at scale.

### "Why store reviews in the database instead of separate files?"

Storing reviews in the database ensures:

- **Atomicity** - Each review is saved immediately (no data loss)
- **Consistency** - Reviews are always associated with the correct events
- **Queryability** - Easy to filter by review status
- **Portability** - Single file contains events and reviews

Separate files would require complex synchronization logic and increase the risk of data loss.

## Future Directions

### Potential Enhancements

**1. Multi-Reviewer Support**

Track reviews from multiple reviewers to compute inter-rater reliability. This would require:

- Reviewer selection dropdown
- Conflict resolution UI (when reviewers disagree)
- Agreement statistics (Cohen's kappa, etc.)

**2. Machine Learning Integration**

Use review decisions to retrain detection algorithms:

- Export accepted/rejected events as training data
- Visualize model confidence vs. review decisions
- Suggest events for review based on uncertainty

**3. Collaborative Review**

Enable multiple reviewers to work on the same dataset:

- Lock events being reviewed (prevent conflicts)
- Real-time synchronization (see others' progress)
- Comment threads (discuss ambiguous events)

**4. Advanced Visualizations**

Add supplementary plots:

- Topographic maps (spatial distribution)
- Time-frequency plots (spectrograms)
- Event property distributions (histograms)

**5. Customizable Workflows**

Allow users to define custom review workflows:

- Multi-stage review (screening → detailed review)
- Conditional logic (if confidence < 0.5, require comment)
- Batch operations (accept all events matching criteria)

## Conclusion

The EEG Review GUI is designed around a core insight: **manual review is a high-throughput, repetitive task that benefits from optimization**. By prioritizing keyboard interaction, performance, and information density, the GUI enables reviewers to process thousands of events efficiently while maintaining consistency and reproducibility.

The architecture balances simplicity (easy to understand and maintain) with performance (handles massive datasets smoothly). Design decisions favor the 80% use case (rapid review of detected events) over edge cases, resulting in a focused, effective tool.

## See Also

- [Tutorial: Your First EEG Event Review Session](../tutorials/eeg-review-gui-tutorial.md) - Learn by doing
- [How-to Guide: Review EEG Events](../how-to/review-eeg-events.md) - Solve specific problems
- [Reference: EEG Review GUI](../reference/eeg-review-gui.md) - Technical specifications
- [Diátaxis Framework](../DIATAXIS_FRAMEWORK.md) - Documentation philosophy
