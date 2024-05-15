(function(){

    var drivers = [];

    function getChartData(selectedDrivers){
        $.ajax({
            method: "POST",
            url: `getracerecords`,
            data: JSON.stringify(selectedDrivers),
            contentType: "application/json; charset=utf-8"
        }).done(function(response){
            Highcharts.chart('raceChartContainer', response);
        });
    };

    function getDrivers() {
        $.ajax({
            method: "GET",
            url: `getdrivers`,
        }).done(function(response){
            drivers = response;
            updateDriversCheckboxes();
            initMultiselect();
        });
    };

    function populateTextSummary(selectedDrivers) {
        var textSummaryContainer = $("#textSummaryContainer");
        textSummaryContainer[0].innerText = "";

        $('<h1>Race Highlights</h1>').appendTo(textSummaryContainer);

        for (var i = 0; i < drivers.length; i++){
            if (selectedDrivers.indexOf(drivers[i].name) === -1 || drivers[i].textSummary == undefined ){
                continue;
            }

            $(`<div class='textSummary'><h2>Driver: ${drivers[i].name}</h2><span class='textSummaryContent'>${drivers[i].textSummary}</div>`).appendTo(textSummaryContainer);
        }
    }

    function initMultiselect() {
        $("#driverSelectionBox").multiSelect();
    }

    function updateDriversCheckboxes() {
        if (drivers && Array.isArray(drivers) && drivers.length > 0){
            var driversSelectionBox = $("#driverSelectionBox")[0];
            for (var i = 0; i < drivers.length; i++){
                $(`<option class='driverSelectionOption' value='${drivers[i].name}' >Driver: ${drivers[i].name}</option>`).appendTo(driversSelectionBox);
            }
        }

        $("#driverSelectionBox").change(function(){
            var selectedDrivers = [];
            $("option.driverSelectionOption:checked").each(function(){
                selectedDrivers.push(this.value);
            });

            getChartData(selectedDrivers);
            populateTextSummary(selectedDrivers);
        });
    }

    getDrivers();
})();