NASCAR Python code for JDS publication. Code is split into two parts: preprocessing of audio-converted driver-pit crew communication, and conversion of preprocessed data into Sankey diagrams of race progress for a given race and driver.

Preprocessing:

    enter directory code-preprocess
    run python on load_json_as_text with any of the JSON race files as input, e.g. % python load_json_as_text.py 1_atlanta.json
    result is a pair of CSV files with import events extracted from the JSON

Sankey visualization:

    ensure flask is installed in your Python distribution
    enter directory code-sankey\SankeyPython2
    run python on app.py, e.g. % python app.py
    flask will provide a localhost port to view the visualization, which should be 127.0.0.1:5000
    enter 127.0.01:5000 into a web browser to see the available races and drivers
    select a race
    select one or more drivers; if the driver(s) have communication text associated with them, a Sankey diagram of important events extracted from the text will be visualized. Sankey diagrams for multiple drivers will overlap one another
