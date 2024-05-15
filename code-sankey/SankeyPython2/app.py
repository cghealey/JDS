from flask import Flask
from flask import request
import flask
import json
import math
import os

app = Flask(__name__)


@app.route("/")
def index():
    return flask.render_template("index.html")


@app.route("/getdrivers")
def getdrivers():
    print( os.getcwd() )
    input_file = open("static/data/input.json")
    input_data = json.load(input_file)
    return input_data["drivers"]


@app.route("/getracerecords", methods=["POST"])
def getracerecords():
    input_file = open("static/data/input.json")
    input_data = json.load(input_file)

    selected_drivers = request.json

    drivers_data = input_data["drivers"]

    drivers = []

    if len(selected_drivers) > 0:
        for driver in drivers_data:
            if driver["name"] in selected_drivers:
                drivers.insert(0, driver)

    # build ranges for conrer speed
    corner_speeds = get_corner_speeds(drivers)

    corner_speeds.sort()

    elements_per_group = math.floor(len(corner_speeds) / 10)
    speed_to_color_map = {}

    for i in range(0, elements_per_group):
        speed_to_color_map.update({corner_speeds[i]: "#e32525"})

    for i in range(elements_per_group, 2 * elements_per_group):
        speed_to_color_map.update({corner_speeds[i]: "#dc0037"})

    for i in range(2 * elements_per_group, 3 * elements_per_group):
        speed_to_color_map.update({corner_speeds[i]: "#d10046"})

    for i in range(3 * elements_per_group, 4 * elements_per_group):
        speed_to_color_map.update({corner_speeds[i]: "#c30053"})

    for i in range(4 * elements_per_group, 5 * elements_per_group):
        speed_to_color_map.update({corner_speeds[i]: "#b10060"})

    for i in range(5 * elements_per_group, 6 * elements_per_group):
        speed_to_color_map.update({corner_speeds[i]: "#9b006a"})

    for i in range(6 * elements_per_group, 7 * elements_per_group):
        speed_to_color_map.update({corner_speeds[i]: "#830072"})

    for i in range(7 * elements_per_group, 8 * elements_per_group):
        speed_to_color_map.update({corner_speeds[i]: "#670078"})

    for i in range(8 * elements_per_group, 9 * elements_per_group):
        speed_to_color_map.update({corner_speeds[i]: "#450b7a"})

    for i in range(9 * elements_per_group, len(corner_speeds)):
        speed_to_color_map.update({corner_speeds[i]: "#101779"})

    # build ranges for understeer gradient
    understeer_values = get_understeer_values(drivers)

    understeer_values.sort()

    understeer_elements_per_group = math.floor(len(understeer_values) / 10)
    understeer_to_color_map = {}

    for i in range(0, understeer_elements_per_group):
        understeer_to_color_map.update({understeer_values[i]: "#b3b3b3"})

    for i in range(
        understeer_elements_per_group, 2 * understeer_elements_per_group
    ):
        understeer_to_color_map.update({understeer_values[i]: "#a2a2a2"})

    for i in range(
        2 * understeer_elements_per_group, 3 * understeer_elements_per_group
    ):
        understeer_to_color_map.update({understeer_values[i]: "#919191"})

    for i in range(
        3 * understeer_elements_per_group, 4 * understeer_elements_per_group
    ):
        understeer_to_color_map.update({understeer_values[i]: "#808080"})

    for i in range(
        4 * understeer_elements_per_group, 5 * understeer_elements_per_group
    ):
        understeer_to_color_map.update({understeer_values[i]: "#707070"})

    for i in range(
        5 * understeer_elements_per_group, 6 * understeer_elements_per_group
    ):
        understeer_to_color_map.update({understeer_values[i]: "#606060"})

    for i in range(
        6 * understeer_elements_per_group, 7 * understeer_elements_per_group
    ):
        understeer_to_color_map.update({understeer_values[i]: "#515151"})

    for i in range(
        7 * understeer_elements_per_group, 8 * understeer_elements_per_group
    ):
        understeer_to_color_map.update({understeer_values[i]: "#424242"})

    for i in range(
        8 * understeer_elements_per_group, 9 * understeer_elements_per_group
    ):
        understeer_to_color_map.update({understeer_values[i]: "#343434"})

    for i in range(9 * understeer_elements_per_group, len(understeer_values)):
        understeer_to_color_map.update({understeer_values[i]: "#262626"})

    chart_data = {}
    chart_data.update({"title": "Race analysis data"})

    series_entry = {}
    series_entry.update({"type": "sankey"})
    series_entry.update({"name": "Race events"})

    series_entry_keys_data = [
        "from",
        "to",
        "weight",
        "color",
        "lapTime",
        "maxCornerSpeed",
        "onThrotleTime",
        "understeerGradient",
    ]
    series_entry.update({"keys": series_entry_keys_data})

    distinct_lap_numbers = get_lap_numbers(drivers)

    lap_to_level_map = {}
    for index, lap in enumerate(distinct_lap_numbers):
        lap_to_level_map.update({lap: index + 1})

    series_entry_data_data = []
    series_entry_nodes_data = []

    for driver in drivers:
        driver_node = {}
        driver_node.update({"id": driver["name"]})
        driver_node.update({"name": "Driver: " + driver["name"]})
        driver_node.update({"level": 0})
        driver_node.update({"color": "red"})
        series_entry_nodes_data.insert(0, driver_node)

        if len( driver[ "events" ] ) == 0:
            continue

        for event in driver["events"]:
            event_node = {}
            event_node.update({"id": driver["name"] + ":" + str(event["lap"])})
            event_node.update({"name": event["name"]})
            event_node.update({"level": lap_to_level_map.get(event["lap"])})
            event_node.update(
                {"color": understeer_to_color_map[event["understeerGradient"]]}
            )
            event_node.update({"lapTime": event["lapTime"]})
            event_node.update({"maxCornerSpeed": event["maxCornerSpeed"]})
            event_node.update({"onThrotleTime": event["onThrotleTime"]})
            event_node.update(
                {"understeerGradient": event["understeerGradient"]}
            )
            series_entry_nodes_data.insert(0, event_node)

        driver_to_first_event_link = []
        first_event = driver["events"][0]
        driver_to_first_event_link.insert(0, driver["name"])
        driver_to_first_event_link.insert(
            1, driver["name"] + ":" + str(first_event["lap"])
        )
        driver_to_first_event_link.insert(2, first_event["lapTime"])
        driver_to_first_event_link.insert(
            3,
            map_speed_and_throttle_to_color(
                speed_to_color_map,
                first_event["maxCornerSpeed"],
                first_event["onThrotleTime"],
            ),
        )
        series_entry_data_data.insert(
            len(series_entry_data_data), driver_to_first_event_link
        )

        for i in range(0, len(driver["events"]) - 1):
            event_to_event_link = []
            event_to_event_link.insert(
                0, driver["name"] + ":" + str(driver["events"][i]["lap"])
            )
            event_to_event_link.insert(
                1, driver["name"] + ":" + str(driver["events"][i + 1]["lap"])
            )
            event_to_event_link.insert(2, driver["events"][i + 1]["lapTime"])
            event_to_event_link.insert(
                3,
                map_speed_and_throttle_to_color(
                    speed_to_color_map,
                    driver["events"][i + 1]["maxCornerSpeed"],
                    driver["events"][i + 1]["onThrotleTime"],
                ),
            )
            series_entry_data_data.insert(
                len(series_entry_data_data), event_to_event_link
            )

    series_entry.update({"colorByPoint": False})
    series_entry.update({"nodes": series_entry_nodes_data})
    series_entry.update({"data": series_entry_data_data})
    series_entry.update(
        {
            "tooltip": {
                "nodeFormat": "<b>{point.name}</b>:<br/>Lap time: <b>{point.lapTime}</b><br/>Max corner speed: <b>{point.maxCornerSpeed}</b><br/>On throttle time: <b>{point.onThrotleTime}</b><br/>Understeer gradient: <b>{point.understeerGradient}</b>"
            }
        }
    )

    series_data = [series_entry]
    chart_data.update({"series": series_data})

    chart_data.update

    return chart_data


def get_understeer_values(drivers_data):
    event_understeer_values = []

    for driver in drivers_data:
        for event in driver["events"]:
            event_understeer_values.insert(0, event["understeerGradient"])

    return event_understeer_values


def get_corner_speeds(drivers_data):
    event_speed_values = []

    for driver in drivers_data:
        for event in driver["events"]:
            event_speed_values.insert(0, event["maxCornerSpeed"])

    return event_speed_values


def get_lap_numbers(drivers_data):
    lap_numbers = []

    for driver in drivers_data:
        for event in driver["events"]:
            lap_numbers.insert(0, event["lap"])

    distinct_lap_numbers = set()
    for lap in lap_numbers:
        distinct_lap_numbers.add(lap)

    laps_list = list(distinct_lap_numbers)
    laps_list.sort()
    return laps_list


def map_speed_and_throttle_to_color(speed_to_color_map, speed, throttle_time):
    current_opacity = 255 * (throttle_time / 100)

    color = speed_to_color_map.get(speed)
    opacityValue = format(int(current_opacity), "02x")

    return color + opacityValue


# if __name__ == "__main__":
#    app.debug = True

app.run()
