let EDUCATE = "educate";

let allData;
let Y_label;

let rec_item;
let rec_item_cor;

let vis1_x = "X1";
let vis2_y = "X1";

let pending_action;
let pending_value;

let pending_rec = false;

function sendInfoToServer(action_type, action_var=null) {
	let timestamp = Date.now();
	console.log("Send: " + action_type + "; " + action_var);
    $.ajaxSetup({
    headers:{'X-CSRFToken': csrf_token},
    });
    $.ajax({
    url: url_log,
    method: 'POST',
    data: {
        'action_type': action_type,
        'action_var': action_var,
        'included_vars': getIncludedVariables().join(","),
        'timestamp': timestamp,
    },
    dataType: 'json',
    success: [function (data) {
        if (data.valid === false) {
            alert("encountered error!");
        } else {
            if (data.rec_item === null) {}
            else if (data.rec_item === EDUCATE) {
                openFullTutorial();
            } else {
                rec_item = data.rec_item;
                rec_item_cor = data.rec_item_cor;
                visualizeReco();
            }
        }
    }]
  });
}

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
    console.log("Vis1: " + userSelectedX);
    vis1_x = userSelectedX;
    $('#vis1-x').val(userSelectedX);
    const myData = generateDataPairFromX(allData[userSelectedX], allData[Y_label]);
    createChart(myData, 'chart-container', userSelectedX, Y_label);
}

function visualize2vars(userSelectedX, userSelectedY) {
    console.log("Vis2: " + userSelectedX + ", " + userSelectedY);
    vis2_y = userSelectedY;
    $('#vis2-y').val(userSelectedY);
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

/* AI */
function visualizeReco() {
    pending_rec = true;

    visualizeVar(rec_item);
    visualize2vars(rec_item, rec_item_cor);

    selectRecInModel();

    let included = varIncludedInModel(rec_item);
    if (included) {
        AIupdateMessage(`Would you like to keep variable ${rec_item} in the model?`)
    } else {
        AIupdateMessage(`Would you like to include variable ${rec_item} in the model?`)
    }
    AIdisplayFeedback();
}

function varIncludedInModel(var_name) {
    let included = false;
    $("#leftValues option").each(function() {
        let var_value = $(this).val();
        if (var_value === var_name) {
            included = true;
        }
    });
    return included;
}

function switchVarInModel(var_name="", to_remove_from, to_include_in) {
    $("#" + to_remove_from + " option[value=" + var_name + "]").remove();
    let opt = document.createElement("option");
    opt.value = var_name;
    opt.innerHTML = var_name;
    let select = $("#" + to_include_in);
    select.append(opt);
    select.val(var_name);
}

function includeVarInModel(var_name="") {
    let included = varIncludedInModel(var_name);
    if (included) {
        console.log("Var " + var_name + " already included");
    }else {
        console.log("Including " + var_name);
        switchVarInModel(rec_item, "rightValues", "leftValues");
    }
}


function excludeVarFromModel(var_name="") {

    let included = varIncludedInModel(var_name)
    if (included) {
        switchVarInModel(rec_item, "leftValues", "rightValues");
    } else {
        console.log("Variable already excluded")
    }
}

function AIaskUserForNewSuggestion() {
    $("#ai-container-btn-feedback").hide();
    $("#ai-container-btn-new").show();
    AIupdateMessage("Would like me to help you exploring the variables?");
}

function AIupdateMessage(message="") {
    $("#recommendation-text").text(message);
}

function AIdisplayFeedback() {
    $("#ai-container-btn-feedback").show();
}

function openFullTutorial() {
    AIupdateMessage("Displaying tutorial...");
    $("#full-tutorial-popup").show();
}


function unSelectEverythingInModel() {
    $("#leftValues option:selected").removeAttr("selected");
    $("#rightValues option:selected").removeAttr("selected");
}

function selectRecInModel() {
    unSelectEverythingInModel();
    let included = varIncludedInModel(rec_item)
    if (included) {
        $("#leftValues").val(rec_item);
    } else {
        $("#rightValues").val(rec_item);
    }
}

function showAiPopupFeedback() {
    $("#ai-popup-feedback").show();
}

function getIncludedVariables() {
    let values = [];
    $("#leftValues option").each(function()
    {
        values.push($(this).val());
    });
    return values;
}

$('#leftValues').change(function(){
    let value = $(this).val();
    console.log("select left " + value);
    if (pending_rec) {
        showAiPopupFeedback();
    }
})

$('#rightValues').change(function(){
    let value = $(this).val();
    console.log("select right " + value);
    if (pending_rec) {
        showAiPopupFeedback();
    }
})

$("#cancel-ignore").click(function() {
    $("#ai-popup-feedback").hide();
    $('#vis1-x').val(rec_item);
    selectRecInModel();
    pending_action = null;
    pending_value = null;
})

$("#confirm-ignore").click(function() {
    if (pending_action === 'vis1-x') {
        visualizeVar(pending_value);
        visualize2vars(pending_value, pending_value);
    }
    $("#ai-popup-feedback").hide();
    AIaskUserForNewSuggestion();
    pending_action = null;
    pending_value = null;
    pending_rec = false;
    sendInfoToServer("ignore", rec_item);
})

$("#close-tutorial").click(function() {
    $("#full-tutorial-popup").hide();
    AIaskUserForNewSuggestion();
    sendInfoToServer('close-tutorial', EDUCATE);
})

$("#accept").click(function(){
    includeVarInModel(rec_item);
    AIaskUserForNewSuggestion();
    pending_rec = false;
    sendInfoToServer("accept", rec_item);
})

$("#refuse").click(function(){
    excludeVarFromModel(rec_item);
    AIaskUserForNewSuggestion();
    pending_rec = false;
    sendInfoToServer("refuse", rec_item);
})

$("#new").click(function(){
    $("#ai-container-btn-new").hide();
    sendInfoToServer("new");
    AIupdateMessage("Let me think...");
})

$("#btnLeft").click(function () {
    if (pending_rec) {
        showAiPopupFeedback();
        return;
    }
    let selectedItem = $("#rightValues option:selected");
    $("#leftValues").append(selectedItem);
    sendInfoToServer("add", selectedItem.val());
});

$("#btnRight").click(function () {
    if (pending_rec) {
        showAiPopupFeedback();
        return;
    }
    let selectedItem = $("#leftValues option:selected");
    $("#rightValues").append(selectedItem);
    sendInfoToServer("remove", selectedItem.val());
});

$('select#vis1-x').click(function(){
    let userSelectedX = $("#vis1-x").val()
    if (pending_rec) {
        pending_action = "vis1-x";
        pending_value = userSelectedX;
        showAiPopupFeedback();
    } else {
        visualizeVar(userSelectedX);
        let userSelectedY = userSelectedX;
        document.getElementById("vis2-y").value = userSelectedY;
        visualize2vars(userSelectedX, userSelectedY);
        sendInfoToServer("vis1-x", userSelectedX);
    }
});

$('select#vis2-y').click(function(){
	const userSelectedX = $("#vis1-x").val();
    const userSelectedY = $("#vis2-y").val();
    visualize2vars(userSelectedX, userSelectedY);
    sendInfoToServer("vis2-y", [userSelectedX, userSelectedY].join(","));
})

$("#button-submit").click(function(){
    console.log("Click on submit");
    $("#popup-submit").show();
    return false;
})

$("#confirm-submit").click(function(){
    let values = getIncludedVariables();
    sendInfoToServer("submit", values.join(','))
    $("#submit-form").submit();
})

$("#cancel-submit").click(function() {
    $("#popup-submit").hide();
})


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
$("#vis1-x").val(labels[0]);
$("#vis2-y").val(labels[0]);
$("#rightValues").val(labels[0]);

// visualizeReco();
console.log(allData);

if (group_id > 0) {
    AIaskUserForNewSuggestion();
}
