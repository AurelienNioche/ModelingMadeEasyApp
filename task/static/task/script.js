// Used to log all actions of the user
function sendInfoToServer(action_type, parameters) {
	let timestamp = Date.now();
	console.log("Send: " + action_type + "; " + parameters);
    $.ajaxSetup({
    headers:{'X-CSRFToken': csrftoken},
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
      }
    }]
  });
}

// click submit to go to modeling_test_after
$(".submitbutton").click(function(){

    let values = [];
    $("#leftValues option").each(function()
    {
        // Add $(this).val() to your list
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
function visualizeVar(userSelectedX, log= true) {
    const myData = generateDataPairFromX(allData[userSelectedX], allData[Y_label]);
    console.log(allData)
    createChart(myData, 'chart-container', userSelectedX, Y_label);
    console.log('ready :)');
    if (log) {
        sendInfoToServer("vis-1", userSelectedX);
    }
}

function visualize2vars(userSelectedXleft, userSelectedXbottom, log= true) {
	const myDataTwo = generateDataPairFromX(allData[userSelectedXbottom],allData[userSelectedXleft]);
    createChart(myDataTwo, 'chart-container-two',userSelectedXbottom, userSelectedXleft);
    console.log('ready :)');
    if (log) {
	    sendInfoToServer("vis-2", [userSelectedXleft, userSelectedXbottom]);
    }
}


$('select#variable-one').click(function(){
    const userSelectedX = document.getElementById("variable-one").value;
	visualizeVar(userSelectedX);
});


/*    DATA VISUALIZATION EVENTS    */
$("select#variable-two").click(function(){
	const userSelectedXleft = document.getElementById("variable-two").value;
    const userSelectedXbottom = document.getElementById("variable-three").value;
    visualize2vars(userSelectedXleft, userSelectedXbottom);
});

$('select#variable-three').click(function(){
	const userSelectedXleft = document.getElementById("variable-two").value;
    const userSelectedXbottom = document.getElementById("variable-three").value;
    visualize2vars(userSelectedXleft, userSelectedXbottom);
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
		populate_select_options(X_labels, "#variable-two");
		populate_select_options(X_labels, "#variable-three");

		// Plot graphs
		visualizeVar(labels[0], false);
		visualize2vars(labels[0], labels[0], false);

		// Select variables in list boxes
		document.getElementById("variable-one").value = labels[0];
		document.getElementById("variable-two").value = labels[0];
		document.getElementById("variable-three").value = labels[0];
});
