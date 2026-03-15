const wavesurfer = WaveSurfer.create({
container:"#waveform",
waveColor:"#90cdf4",
progressColor:"#1c3d5a",
height:120
});

document.getElementById("audioFile").addEventListener("change",(e)=>{

const file = e.target.files[0];

if(file){
wavesurfer.loadBlob(file);
}

});

async function analyzeCry(){

const file = document.getElementById("audioFile").files[0];

if(!file){
alert("Please upload audio first");
return;
}

document.getElementById("status").innerText="Analyzing...";

const formData = new FormData();
formData.append("file",file);

try{

const response = await fetch("/predict",{
method:"POST",
body:formData
});

const data = await response.json();


// Cry Type
document.getElementById("cryType").innerText = data.classification;


// Confidence %
document.getElementById("confValue").innerText =
(data.confidence*100).toFixed(1)+"%";


// Reliability
document.getElementById("confidence").innerText =
data.reliability || "Unknown";


// Risk indicator
if(data.classification.includes("Asphyxia")){
document.getElementById("risk").innerText="High Risk";
}
else{
document.getElementById("risk").innerText="No Risk";
}


// Possible cause (only if baby seems okay)

if(data.classification === "Baby Seems Okay"){

document.getElementById("causeRow").style.display="block";

document.getElementById("cause").innerText =
data.possible_cause || "General Cry";

}
else{

document.getElementById("causeRow").style.display="none";

}


// Status
document.getElementById("status").innerText="Analysis Complete";

}
catch(err){

console.error(err);

document.getElementById("status").innerText="Error analyzing audio";

}

}