# Tutorial: Your First EEG Event Review Session

Welcome! In this tutorial, we'll walk through your first complete event review session using the TurtleWave EEG Review GUI. By the end, you'll have reviewed real EEG events and understand the basic workflow.

!!! note "What you'll learn"
    - How to launch the EEG Review GUI
    - How to load your data files
    - How to navigate through detected events
    - How to accept or reject events
    - How to export your review results

!!! tip "Before you start"
    Make sure you have:
    
    - TurtleWave installed (`pip install turtlewave-hdEEG`)
    - An event database file (`.db` file from event detection)
    - The corresponding EEG data file (`.set` or `.fdt` format)
    - Optional: Sleep stage annotation file (`.xml` format)

## Step 1: Launch the GUI

First, let's start the EEG Review GUI. Open your terminal and run:

```bash
python -m frontend.eeg_review_gui
```

The application window will open. You should see a three-panel interface with a channel selector on the left, a timeline and event list in the middle, and navigation controls on the right.

!!! success "What you should see"
    A window titled "TurtleWave Event Review" with three main panels. The interface is empty because we haven't loaded any data yet.

## Step 2: Load Your Event Database

Now let's load the events you want to review.

1. Click **File** → **Open Database...** in the menu bar
2. Navigate to your event database file (e.g., `subject001_events.db`)
3. Click **Open**

The event list in the middle panel will populate with detected events. You'll see columns showing event type, start time, duration, channel, and other properties.

!!! success "What you should see"
    The middle panel now shows a table filled with events. Each row represents one detected event (spindle, slow wave, etc.).

## Step 3: Load Your EEG Data

To visualize the actual waveforms, we need to load the EEG data file.

1. Click **File** → **Open EEG File...** in the menu bar
2. Navigate to your EEG file (e.g., `subject001.set`)
3. Click **Open**

The GUI will load the data. This may take a moment for large files.

!!! success "What you should see"
    The waveform plot in the middle panel now displays EEG traces for the first event. You'll see multiple channels plotted with the detected event highlighted.

## Step 4: Load Sleep Stage Annotations (Optional)

If you have sleep stage annotations, let's load them to see the hypnogram.

1. Click **File** → **Open Annotation File...** in the menu bar
2. Navigate to your annotation file (e.g., `subject001_annotations.xml`)
3. Click **Open**

The timeline at the top of the middle panel will now show colored bars representing sleep stages (Wake, N1, N2, N3, REM).

!!! success "What you should see"
    A colorful hypnogram appears above the event list, showing the sleep architecture throughout the recording.

## Step 5: Navigate Through Events

Let's explore the detected events.

**Using the toolbar:**

- Click **Next ▶** to move to the next event
- Click **◀ Prev** to go back to the previous event

**Using keyboard shortcuts:**

- Press **→** (right arrow) for next event
- Press **←** (left arrow) for previous event

Notice how the waveform plot updates to show each event as you navigate.

!!! tip "What to observe"
    As you navigate, watch how:
    
    - The selected row in the event table highlights
    - The waveform plot updates to show the new event
    - The event details appear in the right panel

## Step 6: Review Your First Event

Now let's review an event. Look at the waveform plot and decide if the detected event is valid.

**To accept an event:**

- Click **✓ Accept** in the toolbar, OR
- Press **A** on your keyboard

**To reject an event:**

- Click **✗ Reject** in the toolbar, OR
- Press **R** on your keyboard

After you make a decision, the GUI automatically advances to the next event.

!!! success "What you should see"
    The event table updates to show your review decision (a checkmark or X appears in the review column), and the display moves to the next event.

## Step 7: Filter Events by Type

Let's focus on reviewing only one type of event.

1. Look at the **Event Type Filter** section in the right panel
2. Uncheck event types you don't want to see (e.g., uncheck "slow_wave" to see only spindles)
3. Click **Apply Filters**

The event list will update to show only the selected event types.

!!! tip "Efficient reviewing"
    Filtering by event type helps you focus on one category at a time, making your review more consistent.

## Step 8: Adjust the Waveform Display

Let's customize how the waveforms are displayed.

**Change the time window:**

1. Find the **Window Duration** slider in the right panel
2. Drag it to adjust how many seconds of data to display (10-60 seconds)
3. The waveform plot updates immediately

**Toggle filtering:**

1. Find the **Filter Settings** section
2. Check or uncheck **Enable Filter** to toggle bandpass filtering
3. Adjust the **Low** and **High** frequency sliders if needed

!!! note "Why adjust the display?"
    - Longer windows show more context around the event
    - Filtering can help visualize specific frequency components
    - Different event types may benefit from different display settings

## Step 9: Select Channels to Display

Let's choose which EEG channels to visualize.

1. Look at the **Channel Selector** panel on the left
2. Check or uncheck channels to show/hide them
3. The waveform plot updates to show only selected channels

**Quick selection buttons:**

- Click **All** to select all channels
- Click **None** to deselect all channels
- Click **Default** to select common channels (E112, E118, Cz)

!!! tip "Channel selection strategy"
    Start with a few key channels (like Cz, frontal, and occipital) to keep the display clean. Add more channels if you need to verify an event across the scalp.

## Step 10: Review Multiple Events

Now let's review several events in sequence. We'll practice the efficient workflow:

1. Look at the waveform
2. Make a decision (Accept with **A** or Reject with **R**)
3. The GUI automatically moves to the next event
4. Repeat

Try to review at least 10 events to get comfortable with the rhythm.

!!! success "Building momentum"
    You should start to feel a natural flow: observe → decide → advance. This rhythm makes reviewing large datasets manageable.

## Step 11: Jump to Unreviewed Events

If you want to skip events you've already reviewed:

1. Click **Next Unreviewed** in the right panel
2. The GUI jumps to the next event that hasn't been reviewed yet

This is helpful when you're resuming a review session or want to focus only on new events.

## Step 12: Check Your Progress

Let's see how much you've reviewed.

Look at the **Review Statistics** section in the right panel. You'll see:

- Total events in the current view
- Number of events reviewed
- Number accepted
- Number rejected
- Percentage complete

!!! tip "Track your progress"
    These statistics help you estimate how much time remains and maintain consistency in your review decisions.

## Step 13: Export Your Results

Finally, let's save your review decisions.

1. Click **Export** → **Export Reviewed Events...** in the menu bar
2. Choose a location and filename (e.g., `subject001_reviewed.csv`)
3. Click **Save**

The GUI exports a CSV file containing all events with your review decisions.

!!! success "What you've created"
    A CSV file with columns for each event property plus your review decision and timestamp. You can open this in Excel, Python, or R for further analysis.

## What You've Accomplished

Congratulations! You've completed your first EEG event review session. You now know how to:

✅ Launch the GUI and load your data files  
✅ Navigate through detected events efficiently  
✅ Accept or reject events using keyboard shortcuts  
✅ Filter events by type and customize the display  
✅ Select channels to visualize  
✅ Track your progress and export results  

## Next Steps

Now that you're comfortable with the basics, you can:

- **Review larger datasets** - Apply what you've learned to your full study data
- **Customize your workflow** - Explore advanced filtering and display options
- **Learn advanced techniques** - Check out the [How-to Guides](../how-to/review-eeg-events.md) for specific tasks
- **Understand the architecture** - Read the [Explanation](../explanation/eeg-review-gui-architecture.md) to learn how the GUI works

!!! tip "Practice makes perfect"
    The more events you review, the faster and more consistent you'll become. Most reviewers develop a comfortable rhythm after reviewing 50-100 events.
