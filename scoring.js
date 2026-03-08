function v(p1,p2){return{x:p2.x-p1.x,y:p2.y-p1.y}}
function dot(a,b){return a.x*b.x+a.y*b.y}
function mag(a){return Math.hypot(a.x,a.y)}
function angleBetween(a,b,c){const ab=v(b,a);const cb=v(b,c);const d=dot(ab,cb);const m=mag(ab)*mag(cb);if(m===0)return 0;let cos=d/m;if(cos>1)cos=1;if(cos<-1)cos=-1;return Math.acos(cos)*180/Math.PI}
function pick(lm,i){const p=lm[i];return{x:p.x,y:p.y}}
function computeAngles(lm){const LSH=11,LEL=13,LIH=23,LEK=25,LAW=15,LAN=27,LSH2=11;const RSH=12,REL=14,RIH=24,REK=26,RAW=16,RAN=28,RSH2=12
const leftShoulder=angleBetween(pick(lm,LEL),pick(lm,LSH),pick(lm,LIH))
const rightShoulder=angleBetween(pick(lm,REL),pick(lm,RSH),pick(lm,RIH))
const leftElbow=angleBetween(pick(lm,LAW),pick(lm,LEL),pick(lm,LSH2))
const rightElbow=angleBetween(pick(lm,RAW),pick(lm,REL),pick(lm,RSH2))
const leftHip=angleBetween(pick(lm,LEK),pick(lm,LIH),pick(lm,LSH))
const rightHip=angleBetween(pick(lm,REK),pick(lm,RIH),pick(lm,RSH))
const leftKnee=angleBetween(pick(lm,LAN),pick(lm,LEK),pick(lm,LIH))
const rightKnee=angleBetween(pick(lm,RAN),pick(lm,REK),pick(lm,RIH))
return{leftShoulder,rightShoulder,leftElbow,rightElbow,leftHip,rightHip,leftKnee,rightKnee}}
function scoreDiff(d){if(d<20)return 12.5;if(d<40)return 5;return -3}
function scoreAngles(user,ref){const keys=Object.keys(user);let total=0;const per={};const diffs={};for(const k of keys){const d=Math.abs(user[k]-ref[k]);diffs[k]=d;const s=scoreDiff(d);per[k]=s;total+=s}if(total<0)total=0;const percent=Math.min(100,Math.max(0,(total/100)*100));return{total,percent,per,diffs}}
function avgPairs(diffs,keysL,keysR){const a=(diffs[keysL]+diffs[keysR])/2;return a}
window.DanceScoring={computeAngles,scoreAngles,avgPairs} 

