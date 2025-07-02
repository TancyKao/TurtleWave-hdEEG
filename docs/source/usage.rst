Usage
=====

Getting Started
--------------

To launch the TurtleWave GUI:

.. code-block:: python

   from turtlewave_hdEEG.turtlewave_gui import main
   main()

Basic Workflow
-------------

1. **Load Data**: Select your EEG data file and set output directory
2. **Generate Annotations**: Process artifacts, arousals, and sleep stages
3. **Detect Events**: Configure and run spindle or slow wave detection
4. **Analyze Coupling**: Perform phase-amplitude coupling analysis
5. **Review Results**: View detected events and statistics (future feature)