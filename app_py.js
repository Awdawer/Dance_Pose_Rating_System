const userVideo=document.getElementById('userVideo')
const refVideo=document.getElementById('refVideo')
const userCanvas=document.getElementById('userCanvas')
const refCanvas=document.getElementById('refCanvas')
const userCtx=userCanvas.getContext('2d')
const refCtx=refCanvas.getContext('2d')
const userFile=document.getElementById('userFile')
const refFile=document.getElementById('refFile')
const startCam=document.getElementById('startCam')
const stopCam=document.getElementById('stopCam')
const playBtn=document.getElementById('playBtn')
const pauseBtn=document.getElementById('pauseBtn')
const stepBtn=document.getElementById('stepBtn')
const extractBad=document.getElementById('extractBad')
const downloadReport=document.getElementById('downloadReport')
const scoreValue=document.getElementById('scoreValue')
const badFramesGrid=document.getElementById('badFramesGrid')
const API='http://127.0.0.1:8001/api/score'
const POSE_CONNECTIONS=[[11,13],[13,15],[12,14],[14,16],[11,12],[11,23],[12,24],[23,24],[23,25],[25,27],[24,26],[26,28],[27,29],[29,31],[28,30],[30,32]]
let camStream=null
function fitCanvases(){const r1=userVideo.getBoundingClientRect();userCanvas.width=r1.width;userCanvas.height=r1.height;const r2=refVideo.getBoundingClientRect();refCanvas.width=r2.width;refCanvas.height=r2.height}
window.addEventListener('resize',fitCanvases)
let radarChart=null
function initChart(){const ctx=document.getElementById('radarChart').getContext('2d');radarChart=new Chart(ctx,{type:'radar',data:{labels:['Shoulder','Elbow','Hip','Knee'],datasets:[{label:'Score',data:[0,0,0,0],borderColor:'#3b82f6',backgroundColor:'rgba(59,130,246,.15)',pointBackgroundColor:'#3b82f6'}]},options:{responsive:true,maintainAspectRatio:false,scales:{r:{min:0,max:100}},plugins:{legend:{display:false}}})}
initChart()
let processing=false
let badFrames=[]
function drawSkeleton(ctx,landmarks,canvas,diffs){ctx.clearRect(0,0,canvas.width,canvas.height);if(!landmarks||landmarks.length===0)return;const w=canvas.width;const h=canvas.height;const lm=landmarks.map(p=>({x:p.x*w,y:p.y*h}));for(const [a,b] of POSE_CONNECTIONS){const p1=lm[a],p2=lm[b];if(!p1||!p2)continue;ctx.beginPath();ctx.moveTo(p1.x,p1.y);ctx.lineTo(p2.x,p2.y);ctx.lineWidth=2;ctx.strokeStyle='#10b981';ctx.stroke()}for(const p of lm){ctx.beginPath();ctx.arc(p.x,p.y,3,0,Math.PI*2);ctx.fillStyle='#fff';ctx.fill()}if(diffs){const idxs={leftShoulder:11,rightShoulder:12,leftElbow:13,rightElbow:14,leftHip:23,rightHip:24,leftKnee:25,rightKnee:26};for(const k in idxs){const i=idxs[k];const p=lm[i];if(!p)continue;const d=diffs[k]||0;let col='#22c55e';if(d>=40)col='#ef4444';else if(d>=20)col='#f59e0b';ctx.fillStyle=col;ctx.font='12px system-ui';ctx.fillText(d.toFixed(0)+'°',p.x+6,p.y-6)}}}
function updateScoreUI(percent,diffs){scoreValue.textContent=Math.round(percent).toString();const sAvg=(a,b)=>Math.max(0,100-((a+b)/2));const sh=sAvg(diffs.leftShoulder||0,diffs.rightShoulder||0);const el=sAvg(diffs.leftElbow||0,diffs.rightElbow||0);const hp=sAvg(diffs.leftHip||0,diffs.rightHip||0);const kn=sAvg(diffs.leftKnee||0,diffs.rightKnee||0);radarChart.data.datasets[0].data=[sh,el,hp,kn];radarChart.update('none')}
const offUser=document.createElement('canvas');const offRef=document.createElement('canvas');const offUCtx=offUser.getContext('2d');const offRCtx=offRef.getContext('2d')
let lastSend=0
async function processLoop(){if(processing)return;processing=true;fitCanvases();while(processing){const now=performance.now();if(now-lastSend>33){if(userVideo.readyState>=2&&refVideo.readyState>=2){offUser.width=userVideo.videoWidth||640;offUser.height=userVideo.videoHeight||360;offUCtx.drawImage(userVideo,0,0,offUser.width,offUser.height);offRef.width=refVideo.videoWidth||640;offRef.height=refVideo.videoHeight||360;offRCtx.drawImage(refVideo,0,0,offRef.width,offRef.height);const u=offUser.toDataURL('image/jpeg',0.6);const r=offRef.toDataURL('image/jpeg',0.6);try{const res=await fetch(API,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({user_frame:u,ref_frame:r})});const data=await res.json();drawSkeleton(userCtx,data.user_landmarks,userCanvas,data.diffs);drawSkeleton(refCtx,data.ref_landmarks,refCanvas,null);if(data.ok&&data.diffs){updateScoreUI(data.percent||0,data.diffs||{});if(!userVideo.paused){if((data.percent||0)<60){const c=document.createElement('canvas');c.width=userCanvas.width;c.height=userCanvas.height;const cx=c.getContext('2d');cx.drawImage(userCanvas,0,0);badFrames.push({time:userVideo.currentTime,score:data.percent||0,canvas:c});if(badFrames.length>200)badFrames.shift()}}}}catch(e){}}lastSend=now}await new Promise(requestAnimationFrame)}}
function stopProcessing(){processing=false}
userFile.addEventListener('change',e=>{const f=e.target.files&&e.target.files[0];if(!f)return;userVideo.srcObject=null;userVideo.src=URL.createObjectURL(f);userVideo.load()})
refFile.addEventListener('change',e=>{const f=e.target.files&&e.target.files[0];if(!f)return;refVideo.src=URL.createObjectURL(f);refVideo.load()})
startCam.addEventListener('click',async()=>{if(camStream)return;try{camStream=await navigator.mediaDevices.getUserMedia({video:{width:1280,height:720},audio:false});userVideo.srcObject=camStream;userVideo.play();startCam.disabled=true;stopCam.disabled=false}catch(e){}})
stopCam.addEventListener('click',()=>{if(!camStream)return;for(const t of camStream.getTracks())t.stop();camStream=null;userVideo.srcObject=null;startCam.disabled=false;stopCam.disabled=true})
playBtn.addEventListener('click',()=>{userVideo.play();refVideo.play()})
pauseBtn.addEventListener('click',()=>{userVideo.pause();refVideo.pause()})
stepBtn.addEventListener('click',()=>{userVideo.pause();refVideo.pause();const step=1/30;userVideo.currentTime+=step;refVideo.currentTime+=step})
extractBad.addEventListener('click',()=>{badFramesGrid.innerHTML='';const items=[...badFrames].sort((a,b)=>a.score-b.score).slice(0,12);for(const it of items){const url=it.canvas.toDataURL('image/png');const card=document.createElement('div');card.className='badframe-card';const img=document.createElement('img');img.src=url;const meta=document.createElement('div');meta.className='badframe-meta';const t=document.createElement('span');t.textContent=`${it.score.toFixed(0)}%`;const tm=document.createElement('span');tm.textContent=`${it.time.toFixed(2)}s`;meta.appendChild(t);meta.appendChild(tm);card.appendChild(img);card.appendChild(meta);badFramesGrid.appendChild(card)}})
downloadReport.addEventListener('click',async()=>{const {jsPDF}=window.jspdf;const doc=new jsPDF({unit:'pt',format:'a4'});let y=40;doc.setFontSize(18);doc.text('Dance Pose Scoring Report',40,y);y+=24;doc.setFontSize(12);doc.text(`Average score may vary across session runtime.`,40,y);y+=18;const items=[...badFrames].sort((a,b)=>a.score-b.score).slice(0,8);doc.text(`Top ${items.length} bad frames:`,40,y);y+=14;for(const it of items){const png=it.canvas.toDataURL('image/png');doc.text(`${it.score.toFixed(0)}% at ${it.time.toFixed(2)}s`,40,y);y+=12;doc.addImage(png,'PNG',40,y,240,135);y+=150;if(y>760){doc.addPage();y=40}}doc.save('pose_report.pdf')})
Promise.resolve().then(()=>processLoop())
