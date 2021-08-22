// Used to log all actions of the user
function sendInfoToServer(action_type, parameters) {
	date = Date.now();
	console.log(action_type + " " + parameters + " | " + date.toString());
	// TODO: send to server
}



// TODO: To remove
function getRecommendation() {
	// TODO: Get real reco
    const random_number = Math.floor(Math.random() * 8)+1;
    const random_value = "X".concat(random_number.toString());
    // $("#recommendataion").innerHTML = "Variable review recommendation: ".concat(random_value);
    // console.log(":_")
    visualizeReco(random_value);
}



    // click submit to go to modeling_test_after

    $(".submitbutton").click(function(){
        
        const values = [];
        $("#leftValues option").each(function()
        {
            // Add $(this).val() to your list
            values.push($(this).val());
        });
        alert("Submitted: " + values.join(', '));
    })


    $("#btnLeft").click(function () {
        var selectedItem = $("#rightValues option:selected");
        $("#leftValues").append(selectedItem);
		
		sendInfoToServer("add", selectedItem.val());
		// $("#note_test").innerHTML = "Variable review recommendation: "
		getRecommendation(); // TODO: to remove?
        // $("#recommendataion").innerHTML = "Variable review recommendation: ";
    });


    
    $("#btnRight").click(function () {
        var selectedItem = $("#leftValues option:selected");
        $("#rightValues").append(selectedItem);
		
		sendInfoToServer("remove", selectedItem.val());
		// $("#recommendataion").innerHTML = "Variable review recommendation: "
		getRecommendation(); // TODO: to remove?
    });
    


    let m1,m2,n=[],
        mainObj = {};

    let datapairxy =[];
    let datapairxx =[];

    let allData;
	let Y_label;


/*    PLOT GENERATION    */


    function generateDataPairFromX(x, y){
        const data = [];
        for (var i=0; i<x.length;i++){
            data.push({
                x: x[i],
                y: y[i]
            })
        }
        return data;
    };

function createChart(data, containerId, xID, yID){
    var container = document.getElementById(containerId);
    container.innerHTML = '';
    var canvas = document.createElement('canvas');
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



function visualizeVar(userSelectedX) {
    const myData = generateDataPairFromX(allData[userSelectedX], allData[Y_label]);
    console.log(allData)
    const myChart = createChart(myData, 'chart-container', userSelectedX, Y_label);
    console.log('ready :)');
	sendInfoToServer("vis-1", userSelectedX);
}


function visualize2vars(userSelectedXleft, userSelectedXbottom) {
	const myDataTwo = generateDataPairFromX(allData[userSelectedXbottom],allData[userSelectedXleft]);
    const myChartTwo = createChart(myDataTwo, 'chart-container-two',userSelectedXbottom, userSelectedXleft);
    console.log('ready :)');
	sendInfoToServer("vis-2", [userSelectedXleft, userSelectedXbottom]);
}

function visualizeReco(rec_item) {
	$("recommendataion").innerHTML = "Variable review recommendation: ".concat(rec_item);
	const myDataAI = generateDataPairFromX(allData[rec_item], allData[Y_label]);
	const myChartAI = createChart(myDataAI, 'chart-container-AI', rec_item, Y_label);
	console.log('ready :)');
	
	//$('#trybuttontwo').style.visibility="visible";
	$("#trybuttontwo").click(function(){
		visualizeVar(rec_item);
		visualize2vars(rec_item, rec_item);
		sendInfoToServer("accept", rec_item);
		
		document.getElementById("variable-one").value = rec_item;
		document.getElementById("variable-two").value = rec_item;
		document.getElementById("variable-three").value = rec_item;
		
		getRecommendation();
		console.log('ready :)');
		console.log(rec_item);
	})
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



/*    VARIOUS FUNCTIONS   */

function populate_select_options(vars, selectbox_id) {
	for(i in vars)
	{
	   var opt = document.createElement("option");
	   opt.value = vars[i];
	   opt.innerHTML = vars[i];

	   $(selectbox_id).append(opt);
	   //$("#variable-one").append(opt);
	   //$("#variable-two").append(opt);
	   //$("#variable-three").append(opt);
	   
	}
}




/*    LOADING DATA SET    */


//document.querySelector('#trybuttontwo').style.visibility="hidden";

fetch("./data.json")
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
		visualizeVar(labels[0]);
		visualize2vars(labels[0], labels[0]);
		
		
		// Select variables in list boxes
		document.getElementById("variable-one").value = labels[0];
		document.getElementById("variable-two").value = labels[0];
		document.getElementById("variable-three").value = labels[0];
		
		
		// Get recommendation
		getRecommendation(); // TODO: to remove?
		
		
		/*
        const myData = generateDataPairFromX(allData.X1, allData.Y);
        const myChart = createChart(myData, 'chart-container', 'X1', 'Y');
        const myDataTwo = generateDataPairFromX(allData.X1, allData.X1);
        const myChartTwo = createChart(myDataTwo, 'chart-container-two','X1', 'X1');
		*/
    });

