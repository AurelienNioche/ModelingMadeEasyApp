// Used to log all actions of the user
function sendInfoToServer(action_type, parameters) {
	let timestamp = Date.now();
	console.log("Send: " + action_type + "; " + parameters);
    $.ajaxSetup({
    headers:{'X-CSRFToken': csrf_token},
    });
    $.ajax({
    url: url_log,
    method: 'POST',
    data: {
        'action_type': action_type,
        'parameters': parameters,
        'timestamp': timestamp,
    },
    dataType: 'json',
    success: [function (data) {
        if (data.valid === false) {
        alert("encountered error!");
      } else {
          visualizeReco(data.rec_item);
      }
    }]
  });
}

// click submit to go to modeling_test_after
$(".button-submit").click(function(){
    let values = [];
    $("#leftValues option").each(function()
    {
        values.push($(this).val());
    });
    sendInfoToServer("submit", values.join(','))
})

$("#btnLeft").click(function () {
    let selectedItem = $("#rightValues option:selected");
    let values = [];
    selectedItem.each(function()
    {
        values.push($(this).val());
    });
    $("#leftValues").append(selectedItem);
    sendInfoToServer("add", values.join(','));
});

$("#btnRight").click(function () {
    let selectedItem = $("#leftValues option:selected");
    let values = [];
    selectedItem.each(function()
    {
        values.push($(this).val());
    });
    $("#rightValues").append(selectedItem);
    sendInfoToServer("remove", values.join(','));
});

let m1,m2,n=[];
let allData;
let Y_label;

/*    PLOT GENERATION    */
function generateDataPairFromX(x, y){
    const data = [];
    for (let i=0; i<x.length; i++){
        data.push({
            x: x[i],
            y: y[i]
        })
    }
    return data;
}

function createChart(data, containerId, xID, yID){
    let container = document.getElementById(containerId);
    container.innerHTML = '';
    let canvas = document.createElement('canvas');
    container.appendChild(canvas);
    return new Chart(canvas, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Dataset',
                data: data,

            }]
        },
        options: {
            scales:{
                xAxes:[{
                    scaleLabel: {
                      display: true,
                      labelString: xID,
                    }
                }],
                yAxes:[{
                    scaleLabel: {
                      display: true,
                      labelString: yID
                    }
                }]
            }
        }
    });
}


// Update graphs based on user selection
function visualizeVar(userSelectedX) {
    const myData = generateDataPairFromX(allData[userSelectedX], allData[Y_label]);
    createChart(myData, 'chart-container', userSelectedX, Y_label);
}

function visualize2vars(userSelectedX, userSelectedY) {
	const myDataTwo = generateDataPairFromX(allData[userSelectedX],allData[userSelectedY]);
    createChart(myDataTwo, 'chart-container-two',userSelectedX, userSelectedY);
}


$('select#variable-one').click(function(){
    const userSelectedX = document.getElementById("variable-one").value;
	visualizeVar(userSelectedX);
	sendInfoToServer("vis-1", userSelectedX);
});


/*    DATA VISUALIZATION EVENTS    */
$("select#var-c-x").click(function(){
	const userSelectedX = document.getElementById("var-c-x").value;
    const userSelectedY = document.getElementById("var-c-y").value;
    visualize2vars(userSelectedX, userSelectedY);
    sendInfoToServer("vis-2-x", [userSelectedX, userSelectedY].join(","));
});

$('select#var-c-y').click(function(){
	const userSelectedX = document.getElementById("var-c-x").value;
    const userSelectedY = document.getElementById("var-c-y").value;
    visualize2vars(userSelectedX, userSelectedY);
    sendInfoToServer("vis-2-y", [userSelectedX, userSelectedY].join(","));
})


/*  populate selection options   */
function populate_select_options(vars, selectbox_id) {
	for(let i in vars)
	{
	   let opt = document.createElement("option");
	   opt.value = vars[i];
	   opt.innerHTML = vars[i];
	   $(selectbox_id).append(opt);
	}
}


/*    LOADING DATA SET    */
fetch(data_file)
    .then(response => response.json())
    .then(data=>{
		// Read data
        allData = data;
		const labels = Object.keys(data);
		const X_labels = labels.slice(0,-1);
		Y_label = labels[labels.length - 1];

		// Populate select options
		populate_select_options(X_labels, "#rightValues");
		populate_select_options(X_labels, "#variable-one");
		populate_select_options(X_labels, "#var-c-x");
		populate_select_options(X_labels, "#var-c-y");

		// Plot graphs
		visualizeVar(labels[0]);
		visualize2vars(labels[0], labels[0]);

		// Select variables in list boxes
		document.getElementById("variable-one").value = labels[0];
		document.getElementById("var-c-x").value = labels[0];
		document.getElementById("var-c-y").value = labels[0];

		visualizeReco(init_rec_item);
});

/* AI */
function visualizeReco(rec_item) {
    if (rec_item === null) {
        return;
    }
    document.getElementById("recommendation-text").innerText = "Variable review recommendation: ";
    document.getElementById("recommendation-value").innerText = rec_item;
    console.log("Variable review recommendation: ".concat(rec_item));
	const myDataAI = generateDataPairFromX(allData[rec_item], allData[Y_label]);
	createChart(myDataAI, 'chart-container-ai', rec_item, Y_label);
	console.log('ai ready');
}

$("#button-ai").click(function(){
    let rec_item = document.getElementById("recommendation-value").innerText;
    console.log("rec_item is:".concat(rec_item));
    visualizeVar(rec_item);
    visualize2vars(rec_item, rec_item);

    sendInfoToServer("accept", rec_item);

    document.getElementById("variable-one").value = rec_item;
    document.getElementById("var-c-x").value = rec_item;
    document.getElementById("var-c-y").value = rec_item;
})


