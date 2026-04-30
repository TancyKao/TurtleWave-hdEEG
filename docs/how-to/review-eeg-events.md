# How-to Guide: Review EEG Events

This guide provides practical solutions for specific tasks when reviewing EEG events with the TurtleWave GUI.

## Filter Events by Multiple Criteria

**Problem:** You need to review only specific types of events from certain sleep stages.

**Solution:**

1. Open the **Event Type Filter** section in the right panel
2. Check only the event types you want (e.g., only "spindle")
3. Open the **Sleep Stage Filter** section
4. Check only the sleep stages you want (e.g., only "N2" and "N3")
5. Click **Apply Filters**

The event list updates to show only events matching all selected criteria.

!!! tip
    Filters are cumulative - events must match ALL selected criteria to appear.

## Review Only High-Confidence Events

**Problem:** You want to focus on events with high detection confidence scores.

**Solution:**

1. Locate the **Confidence Threshold** slider in the right panel
2. Drag the slider to set your minimum confidence (e.g., 0.7 for 70%)
3. Click **Apply Filters**

Only events with confidence scores above your threshold will appear.

!!! example
    Setting confidence to 0.8 shows only the top 20% most confident detections, reducing false positives.

## Resume a Previous Review Session

**Problem:** You need to continue reviewing where you left off.

**Solution:**

1. Load your database file (**File** → **Open Database...**)
2. Load your EEG and annotation files as usual
3. Click **Next Unreviewed** in the right panel

The GUI jumps to the first event without a review decision.

!!! note
    Review decisions are saved in the database immediately, so you can safely close and reopen the GUI without losing progress.

## Review Events from a Specific Channel

**Problem:** You want to review only events detected on frontal channels.

**Solution:**

1. Open the **Channel Filter** section in the right panel
2. Enter channel names or patterns (e.g., "E11, E12, E18, E19" or "Fp1, Fp2")
3. Click **Apply Filters**

The event list shows only events from the specified channels.

!!! tip
    Use the channel selector on the left to visualize multiple channels while filtering events from specific channels.

## Compare Events Across Detection Methods

**Problem:** You want to see how different detection algorithms performed.

**Solution:**

1. Open the **Method Filter** section in the right panel
2. Check the detection methods you want to compare
3. Click **Apply Filters**
4. Sort the event table by clicking the **Method** column header

Events from different methods appear grouped together for easy comparison.

!!! example
    Compare "wavelet" vs "bandpass" spindle detection by filtering for both methods and reviewing events side-by-side.

## Adjust Display for Different Event Types

**Problem:** Spindles and slow waves need different display settings for optimal visualization.

**Solution:**

**For spindles (11-16 Hz):**

1. Set **Window Duration** to 10-15 seconds
2. Enable **Filter**
3. Set **Low** frequency to 11 Hz
4. Set **High** frequency to 16 Hz

**For slow waves (0.5-4 Hz):**

1. Set **Window Duration** to 30-60 seconds
2. Enable **Filter**
3. Set **Low** frequency to 0.5 Hz
4. Set **High** frequency to 4 Hz

!!! tip
    Save time by reviewing all events of one type before switching to another type, so you don't need to constantly adjust settings.

## Export Only Accepted Events

**Problem:** You want to export only the events you accepted, not rejected ones.

**Solution:**

1. Open the **Review Status Filter** section
2. Check **Reviewed Only**
3. Click **Apply Filters**
4. Click **Export** → **Export Reviewed Events...**
5. Save the file

The exported CSV contains only events with review decisions. You can then filter the CSV file in Excel or Python to keep only accepted events.

!!! note
    The exported CSV includes a `review_decision` column with values "accept" or "reject" for easy filtering.

## Quickly Review Events in a Time Window

**Problem:** You need to review events only from a specific time period (e.g., first 2 hours of sleep).

**Solution:**

1. Click on the timeline plot at the start of your desired time window
2. The GUI jumps to the first event at that time
3. Review events using keyboard shortcuts (**A** for accept, **R** for reject)
4. Continue until you reach events outside your time window

!!! tip
    The timeline is clickable - click anywhere to jump to events near that time point.

## Handle Overlapping Events

**Problem:** Multiple events are detected at the same time on different channels.

**Solution:**

1. Use the **Channel Selector** on the left to show all relevant channels
2. Review the first event
3. Press **→** to advance to the next event
4. If it's an overlapping event, you'll see it on a different channel at the same time
5. Make independent decisions for each channel

!!! note
    Each event is reviewed independently, even if they overlap in time. This allows channel-specific quality assessment.

## Batch Review Similar Events

**Problem:** You have many similar-looking events and want to review them quickly.

**Solution:**

1. Filter to show only one event type
2. Sort by a relevant property (click column header, e.g., "Duration" or "Amplitude")
3. Use keyboard shortcuts exclusively:
   - **A** to accept
   - **R** to reject
   - **→** to skip without reviewing
4. Maintain a steady rhythm: observe → decide → advance

!!! tip
    Most experienced reviewers can process 100-200 events per hour using keyboard shortcuts and consistent criteria.

## Verify Events Across Multiple Channels

**Problem:** You want to confirm an event is present on multiple channels before accepting.

**Solution:**

1. In the **Channel Selector**, select multiple channels in the same region (e.g., all frontal channels)
2. Review the event - all selected channels appear in the waveform plot
3. Look for the event signature across channels
4. Accept only if the event is clearly visible on multiple channels

!!! example
    For spindles, look for the characteristic waxing-and-waning pattern on at least 2-3 nearby channels before accepting.

## Review Events by Frequency Band

**Problem:** You want to review only slow spindles (11-13 Hz) separately from fast spindles (13-16 Hz).

**Solution:**

1. Open the **Frequency Band Filter** section
2. Select the frequency band you want (e.g., "11-13 Hz")
3. Click **Apply Filters**
4. Review all events in that band
5. Change to the next band and repeat

!!! note
    Frequency bands are automatically detected from your database. If you don't see this filter, your events may not have frequency band information.

## Add Comments to Events

**Problem:** You want to note something specific about an event (e.g., "artifact present" or "borderline").

**Solution:**

1. Review the event normally (**A** or **R**)
2. The **Comments** field in the right panel becomes active
3. Type your comment
4. Press **Enter** or click outside the field to save

Comments are saved with the review decision and included in exported files.

!!! tip
    Use consistent comment keywords (e.g., "artifact", "borderline", "excellent") to make filtering easier later.

## Review Events in Random Order

**Problem:** You want to avoid bias from reviewing events in chronological order.

**Solution:**

1. Load your database and data files
2. Click the **Event ID** column header in the event table
3. Click it again to randomize the sort order
4. Review events in the new order

!!! warning
    The GUI doesn't have a built-in randomize function, but you can pre-process your database to add a random sort column before loading.

## Export Review Statistics

**Problem:** You need to report inter-rater reliability or review completion rates.

**Solution:**

The **Review Statistics** panel shows:

- Total events
- Reviewed count
- Accepted count
- Rejected count
- Completion percentage

Take a screenshot or manually record these values. For programmatic access, query the database directly:

```python
import sqlite3
conn = sqlite3.connect('your_events.db')
cursor = conn.cursor()
cursor.execute("""
    SELECT 
        COUNT(*) as total,
        SUM(CASE WHEN review_decision IS NOT NULL THEN 1 ELSE 0 END) as reviewed,
        SUM(CASE WHEN review_decision = 'accept' THEN 1 ELSE 0 END) as accepted,
        SUM(CASE WHEN review_decision = 'reject' THEN 1 ELSE 0 END) as rejected
    FROM events
""")
stats = cursor.fetchone()
print(f"Total: {stats[0]}, Reviewed: {stats[1]}, Accepted: {stats[2]}, Rejected: {stats[3]}")
```

## Troubleshooting

### Events Not Loading

**Problem:** The event list is empty after loading the database.

**Solution:**

1. Check that your database file is not corrupted (try opening it with a SQLite browser)
2. Verify the database contains events: `SELECT COUNT(*) FROM events`
3. Check if filters are too restrictive - click **Clear Filters**

### Waveforms Not Displaying

**Problem:** The waveform plot is blank even though events are listed.

**Solution:**

1. Verify you loaded the correct EEG file (**File** → **Open EEG File...**)
2. Check that the EEG file path matches the one used during event detection
3. Ensure the EEG file format is supported (.set, .fdt, .edf)
4. Check the console for error messages

### GUI Running Slowly

**Problem:** The GUI is laggy when navigating between events.

**Solution:**

1. Reduce the number of displayed channels (use **Channel Selector**)
2. Decrease the **Window Duration** to show less data
3. Filter to show fewer events (use event type or confidence filters)
4. Close other applications to free up memory

### Keyboard Shortcuts Not Working

**Problem:** Pressing A or R doesn't review events.

**Solution:**

1. Click on the waveform plot or event table to ensure the GUI has focus
2. Check that you're not typing in a text field (click outside any text boxes)
3. Restart the GUI if shortcuts remain unresponsive

## See Also

- [Tutorial: Your First EEG Event Review Session](../tutorials/eeg-review-gui-tutorial.md) - Learn the basics
- [Reference: GUI Components](../reference/gui-components.md) - Detailed component documentation
- [Explanation: Review GUI Architecture](../explanation/eeg-review-gui-architecture.md) - Understand how it works
