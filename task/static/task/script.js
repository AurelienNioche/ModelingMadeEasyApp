// Used to log all actions of the user
function sendInfoToServer(action_type, parameters=null) {
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
            visualizeReco(data.rec_item, data.rec_item_cor);
        }
    }]
  });
}


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
		populate_select_options(X_labels, "#vis1-x");
		populate_select_options(X_labels, "#vis2-y");

		// Plot graphs
		visualizeVar(labels[0]);
		visualize2vars(labels[0], labels[0]);

		// Select variables in list boxes
        $("vis1-x").val(labels[0]);
		$("vis2-y").val(labels[0]);

		visualizeReco(init_rec_item);
});

/* AI */
function visualizeReco(rec_item, rec_item_cor=null) {
    if (rec_item === -1) {
    } else if (rec_item === "educate") {
        openFullTutorial();
    } else {
        rec_item = `X${rec_item}`;
        rec_item_cor = `X${rec_item_cor}`;

        visualizeVar(rec_item);
        visualize2vars(rec_item, rec_item_cor);

        $('#vis1-x').val(rec_item);
        $('#vis2-y').val(rec_item_cor);

        AIupdateMessage("Variable review recommendation: ", rec_item, "Do you think it is a good suggestion?");
        AIdisplayFeedback();
    }
}

function moveVarInModel(var_name="") {
    console.log("Looking for " + var_name)
    let include = false;
    $("#rightValues option").each(function() {
        if ($(this).val() === var_name) {
            include = true;
        }
    });
    console.log("include" + include);
    let to_remove_from;
    let to_include_in;
    if (include) {
        to_remove_from = "rightValues";
        to_include_in = "leftValues";
    } else {
        to_remove_from = "leftValues";
        to_include_in = "rightValues";
    }
    $("#" + to_remove_from + " option[value=" + var_name + "]").remove();
    let opt = document.createElement("option");
    opt.value = var_name;
    opt.innerHTML = var_name;
    let select = $("#" + to_include_in);
    select.append(opt);
    select.val(var_name);
}



function AIaskUserForNewSuggestion() {
    document.getElementById("ai-container-btn-feedback").style.display = "none";
    document.getElementById("ai-container-btn-new").style.display = "block";
    AIupdateMessage("Would you like me to provide a suggestion?");
}

function AIupdateMessage(message="", value="", after_value="") {
    $("#recommendation-text").text(message);
    $("#recommendation-value").text(value);
    $("#recommendation-text-after-value").text(after_value);
}

function AIdisplayFeedback() {
    document.getElementById("ai-container-btn-new").style.display = "none";
    document.getElementById("ai-container-btn-feedback").style.display = "block";
}


function openFullTutorial() {
    AIupdateMessage("Displaying tutorial...");
    $("#full-tutorial-popup").show();
}

function closeFullTutorial() {
    $("#full-tutorial-popup").hide();
    sendInfoToServer('close-tutorial');
}

function cancelIgnoreAI() {
    $("#ai-popup-feedback").hide();
}

function confirmIgnoreAI() {
    $("#ai-popup-feedback").hide();
    AIaskUserForNewSuggestion();
}

function aiIsWaiting() {
    return $("#recommendation-value").text() !== "";
}

function showAiPopupFeedback() {
    $("#ai-popup-feedback").show();
}

$("#accept").click(function(){
    let rec_item = $("#recommendation-value").text();
    moveVarInModel(rec_item);
    AIaskUserForNewSuggestion();
    sendInfoToServer("accept", rec_item);
})

$("#refuse").click(function(){
     let rec_item = $("#recommendation-value").text();
    AIaskUserForNewSuggestion();
    sendInfoToServer("refuse", rec_item);
})

$("#new").click(function(){
    sendInfoToServer("new");
})

$('select#vis1-x').click(function(){
    if (aiIsWaiting()) {
        showAiPopupFeedback();
        return;
    }
    const userSelectedX = document.getElementById("vis1-x").value;
	visualizeVar(userSelectedX);
	let userSelectedY = userSelectedX;
	document.getElementById("vis2-y").value = userSelectedY;
	visualize2vars(userSelectedX, userSelectedY);
	sendInfoToServer("vis1-x", userSelectedX);
});


/*    DATA VISUALIZATION EVENTS    */
// $("select#var-c-x").click(function(){
// 	const userSelectedX = document.getElementById("va-x").value;
//     const userSelectedY = document.getElementById("var-c-y").value;
//     visualize2vars(userSelectedX, userSelectedY);
//     sendInfoToServer("vis-2-x", [userSelectedX, userSelectedY].join(","));
// });

$('select#vis2-y').click(function(){
    if (aiIsWaiting()) {
        showAiPopupFeedback();
        return;
    }
	const userSelectedX = document.getElementById("vis1-x").value;
    const userSelectedY = document.getElementById("vis2-y").value;
    visualize2vars(userSelectedX, userSelectedY);
    sendInfoToServer("vis2-y", [userSelectedX, userSelectedY].join(","));
})

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
    if (aiIsWaiting()) {
        showAiPopupFeedback();
        return;
    }
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
    if (aiIsWaiting()) {
        showAiPopupFeedback();
        return;
    }
    let selectedItem = $("#leftValues option:selected");
    let values = [];
    selectedItem.each(function()
    {
        values.push($(this).val());
    });
    $("#rightValues").append(selectedItem);
    sendInfoToServer("remove", values.join(','));
});


if (group_id > 0) {
    AIaskUserForNewSuggestion();
}
