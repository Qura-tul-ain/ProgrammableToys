var canvas = document.getElementById("canvas");
canvas.width = 730;
canvas.height = 415;

const c = canvas.getContext('2d');
var stage_count;
stage_count=0
// Object that checks if key is pressed
let keyPressed = {
	left: false,
	right: false,
	up: false,
	down: false
}
function setstagefromblockly(stage){
	
	stage_count=stage;
}

// Goal object
const Goal = {
	x: canvas.width - 94,
	y: canvas.height - 70,
	width: 40,
	height: 40,
	draw: function(){
		c.beginPath();
		c.rect(this.x, this.y, this.width, this.height);
		c.fillStyle = "#13E864";
		c.fill();
		c.stroke();
		c.closePath();
	}
}
// positions of stop and chlild images for checks.
// 1st stage 
const S1 = {
	x: 120,
	y:20,
	width: 40,
	height: 40,

}

const S2 = {
	x: 120,
	y: 75,
	width: 40,
	height: 40,

}
const S3 = {
	x: 120,
	y: 185,
	width: 40,
	height: 40,

}
const S4 = {
	x: 120,
	y: 235,
	width: 40,
	height: 40,

}
const S5 = {
	x: 120,
	y: 290,
	width: 40,
	height: 40,

}
const S6 = {
	x: 320,
	y: 20,
	width: 40,
	height: 40,

}
const S7 = {
	x: 320,
	y: 235,
	width: 40,
	height: 40,

}
const S8 = {
	x: 320,
	y: 345,
	width: 40,
	height: 40,

}
const S9 = {
	x: 625,
	y: 185,
	width: 40,
	height: 40,

}
const S10 = {
	x: 625,
	y: 290,
	width: 40,
	height: 40,

 }
const G1 = {
	x: 120,
	y: 130,
	width: 40,
	height: 40,

}
const G2 = {
	x: 520,
	y: 345,
	width: 40,
	height: 40,

}
const G3 = {
	x: 620,
	y: 75,
	width: 40,
	height: 40,

}
const G4 = {
	x: 420,
	y: 235,
	width: 40,
	height: 40,

}
const G5 = {
	x: 636,
	y: 345,
	width: 40,
	height: 40,

}

// for 2nd stage 
const se1 = {
	x: 150,
	y: 90,
	width: 40,
	height: 40,

}
const se2 = {
	x: 35,
	y: 220,
	width: 40,
	height: 40,

}
const se3 = {
	x: 270,
	y: 220,
	width: 40,
	height: 40,

}
const se4 = {
	x: 270,
	y: 155,
	width: 40,
	height: 40,

}
const se5 = {
	x: 270,
	y: 280,
	width: 40,
	height: 40,

}
const se6 = {
	x: 150,
	y: 280,
	width: 40,
	height: 40,

}
const se7 = {
	x: 500,
	y: 345,
	width: 40,
	height: 40,

}
const se8 = {
	x: 500,
	y: 220,
	width: 40,
	height: 40,

}
const se9 = {
	x: 500,
	y: 90,
	width: 40,
	height: 40,

}
const se10 = {
	x: 610,
	y: 90,
	width: 40,
	height: 40,

}
const se11 = {
	x: 610,
	y: 220,
	width: 40,
	height: 40,

}
const se12 = {
	x: 615,
	y: 25,
	width: 40,
	height: 40,

}

const seGoal = {
	x: 620,
	y: 340,
	width: 40,
	height: 40,

}

// Ball object 
// 1st stage 
function Ball(x, y, radius, dx, dy){
	this.x = x;
	this.y = y;
	this.radius = radius;
	this.dx = dx;
	this.dy = dy;

	this.draw = function(){
		c.beginPath();
		c.arc(this.x, this.y, this.radius, 0, Math.PI * 2, false);
		c.fillStyle = "#FF225A";
		c.fill();
		c.stroke();
		c.closePath();
	    console.log("hahaha");
	}

	this.moveRight = function(){

		   if(this.x + radius < canvas.width){
				this.x += this.dx +93;
		   }

		  this.draw();
	}
	
	this.moveDown = function(){
		
			if(this.y + radius < canvas.height){
				this.y += this.dy +48;
			}
			
		   this.draw();
		
	}
	
	
	this.moveUp = function(){
		
			 if(this.y - radius > 0){
				this.y -= this.dy +47;
			 }
			
		   this.draw();
		
	}
	
	
	
	this.moveLeft = function(){
		
			if(this.x - radius > 0){
				this.x -= this.dx +90;
			}
			
		   this.draw();
		
	}
	// for 2nd stage 
	
	
	this.moveSecRight = function(){

		   if(this.x + radius < canvas.width){
				this.x += this.dx +118;
		   }

		  this.draw();
	}
	
	this.moveSecDown = function(){
		
			if(this.y + radius < canvas.height){
				this.y += this.dy +57;
			}
			
		   this.draw();
		
	}
	
	
	this.moveSecUp = function(){
		
			 if(this.y - radius > 0){
				this.y -= this.dy +63;
			 }
			
		   this.draw();
		
	}
	
	
	
	this.moveSecLeft = function(){
		
			if(this.x - radius > 0){
				this.x -= this.dx +116;
			}
			
		   this.draw();
		
	}
	
}

// functions for movemant of balls when stages are interconnected.
function callfunctionDown(){
	//console.log("infunctionnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn",stage_count);
	// ball.moveDown();
	if(stage_count ==1){
			
			  ball.moveDown();
	       
	          console.log("courjhrhff",stage_count);
			
	  }
	  else if(stage_count==2){
		  secondball.moveSecDown();
		  console.log("button",stage_count);

	  }
	  else if(stage_count==3){
		

	  }
}
function callfunctionUp(){
	if(stage_count ==1){
			 // drawBoard();
			  ball.moveUp();
	         // stage_count=goalCheck(stage_count);
			 // document.location.reload();
	          console.log(stage_count);
			
		  }
		  else if(stage_count==2){
			   secondball.moveSecUp();
			   console.log(stage_count);

		  }
		  else if(stage_count==3){
			

		  }
}
function callfunctionLeft(){
	if(stage_count ==1){
			 // drawBoard();
			  ball.moveLeft();
	         // stage_count=goalCheck(stage_count);
			 // document.location.reload();
	          console.log(stage_count);
			
		  }
		  else if(stage_count==2){
			  secondball.moveSecLeft();
			  console.log(stage_count);

		  }
		  else if(stage_count==3){
			

		  }
}
function callfunctionRight(){
	if(stage_count ==1){
			 // drawBoard();
			  ball.moveRight();
	         // stage_count=goalCheck(stage_count);
			 // document.location.reload();
	          console.log(stage_count);
			
		  }
		  else if(stage_count==2){
			  secondball.moveSecRight();

		  }
		  else if(stage_count==3){
			

		  }
}

// functions for stages
function movefunctionDown(){
	if(stage_count ==1){
			 // drawBoard();
			  ball.moveRight();
	         // stage_count=goalCheck(stage_count);
			 // document.location.reload();
	          console.log(stage_count);
			
		  }
		  else if(stage_count==2){
			  secondball.moveSecRight();

		  }
		  else if(stage_count==3){
			

		  }
}




// Wall object
function Wall(x, y, width, height){
	this.x = x;
	this.y = y;
	this.width = width;
	this.height = height;

	this.draw = function(){
		c.beginPath();
		c.rect(this.x, this.y, this.width, this.height);
		c.fillStyle = "black";
		c.fill();
		c.closePath();
	}
}


// Create walls and save it in an array
// let wallArray = [
// new Wall(0, 100, canvas.width - 150, 5),
// new Wall(150, 200, canvas.width, 5),
// new Wall(150, 200, 5, 125),
// new Wall(250, canvas.height - 125, 5, 125),
// new Wall(350, 200, 5, 125),
// new Wall(450, canvas.height - 125, 5, 125),
// new Wall(550, 200, 5, 125),
// new Wall(650, canvas.height - 125, 5, 125)
// ];

// Create the ball using the object 
console.log("balll")
let ball = new Ball(50, 35, 20, 7, 7);
let secondball=new Ball(50,35,20,7,7);

// Check function if ball touchs walls
function RectCircleColliding(circle,rect){
    let distX = Math.abs(circle.x - rect.x-rect.width/2);
    let distY = Math.abs(circle.y - rect.y-rect.height/2);

    if (distX > (rect.width/2 + circle.radius)) { return false; }
    if (distY > (rect.height/2 + circle.radius)) { return false; }

    if (distX <= (rect.width/2)) { return true; } 
    if (distY <= (rect.height/2)) { return true; }

    let dx=distX-rect.width/2;
    let dy=distY-rect.height/2;
    return (dx*dx+dy*dy<=(circle.radius*circle.radius));
}

// function wallsCheck(){
	// if(RectCircleColliding(ball,imagesArr[0])){
		// alert("You lost. Try Again!");
		// document.location.reload();
	// }
	// if(RectCircleColliding(ball, imagesArr[1])){
		// alert("You lost. Try Again!");
		// document.location.reload();
	// }
	// if(RectCircleColliding(ball, imagesArr[2])){
		// alert("You lost. Try Again!");
		// document.location.reload();
	// }
	// if(RectCircleColliding(ball, imagesArr[3])){
		// alert("You lost. Try Again!");
		// document.location.reload();
	// }
	// if(RectCircleColliding(ball, imagesArr[4])){
		// alert("You lost. Try Again!");
		// document.location.reload();
	// }
	// if(RectCircleColliding(ball, imagesArr[5])){
		// alert("You lost. Try Again!");
		// document.location.reload();
	// }
	// if(RectCircleColliding(ball, wallArray[6])){
		// alert("You lost. Try Again!");
		// document.location.reload();
	// }
	// if(RectCircleColliding(ball, wallArray[7])){
		// alert("You lost. Try Again!");
		// document.location.reload();
	// }
// }

// Check function if ball touchs goal square
// for 1st stage 
function goalCheck(count){
	if(RectCircleColliding(ball, S1)){
		if(alert('Alert For your User!')){}
        else    window.location.reload();
		//setTimeout( function ( ) { alert( "moo" ); }, 1000000 );
		// if(confirm('Successful Message')){
        // window.location.reload();  

		 //window.location.reload();
	return count
		
		
	}
	if(RectCircleColliding(ball, S2)){
		if(alert('Alert For your User!')){}
		 else    window.location.reload();

        return count
		// document.location.reload();
	}
	if(RectCircleColliding(ball, S3)){
		if(alert('Alert For your User!')){}
		 else    window.location.reload();

		return count
	
		
	}
	if(RectCircleColliding(ball, S4)){
		 if(alert('Alert For your User!')){}
		 else    window.location.reload();

		return count
	
	}
	if(RectCircleColliding(ball, S5)){
		 if(alert('Alert For your User!')){}
		 else    window.location.reload();
	
		return count
		
	}
	if(RectCircleColliding(ball, S6)){
         if(alert('Alert For your User!')){}
		 else    window.location.reload();
		return count
	
	}
	if(RectCircleColliding(ball, S7)){
		 if(alert('Alert For your User!')){}
		 else    window.location.reload();

		return count
		
	}
	if(RectCircleColliding(ball, S8)){
		 if(alert('Alert For your User!')){}
		 else    window.location.reload();
		  return count
		
	}
	if(RectCircleColliding(ball, S9)){
		 if(alert('Alert For your User!')){}
		 else    window.location.reload();
		return count
		
	}
	if(RectCircleColliding(ball, S10)){
		 if(alert('Alert For your User!')){}
		 else    window.location.reload();
		return count
		
	}
	if(RectCircleColliding(ball, G1)){
	     if(alert('Alert For your User!')){}
		 else    window.location.reload();
		return count
	    		
		// document.location.reload();
	}
	// if(RectCircleColliding(ball, G2)){
		// alert("YOU SHARED");
		//document.location.reload();
	// }
	if(RectCircleColliding(ball, G3)){
		 if(alert('Alert For your User!')){}
		 else    window.location.reload();
		return count
		
	}
	if(RectCircleColliding(ball, G5)){
	     // if(alert('Alert For your User!')){count=count+1}
		 // else   { 
		 count=count+1
		 stage_count=stage_count+1
	 
	   	return count
	}
}

// checkes for 2nd stage 
 function goalCheckForSecond(count){
	if(RectCircleColliding(secondball, se1)){
		alert("YOU LOST!");
		
		document.location.reload();
		return count
	}
	if(RectCircleColliding(secondball, se2)){
		alert("YOU LOST");
	
        return count
		document.location.reload();
	}
	if(RectCircleColliding(secondball, se3)){
		alert("YOU LOST");
	
		return count
		document.location.reload();
		
	}
	if(RectCircleColliding(secondball, se4)){
		alert("YOU LOST");
		
		return count
		document.location.reload();
	}
	if(RectCircleColliding(secondball, se5)){
		alert("YOU LOST");
	
		return count
		document.location.reload();
	}
	if(RectCircleColliding(secondball, se6)){
		alert("YOU LOST");
	
		return count
		document.location.reload();
	}
	if(RectCircleColliding(secondball, se7)){
		alert("YOU LOST");
	
		return count
		document.location.reload();
	}
	if(RectCircleColliding(secondball, se8)){
		alert("YOU LOST");
	
		return count
		document.location.reload();
	}
	if(RectCircleColliding(secondball, se9)){
		alert("YOU LOST");
	
		return count
		document.location.reload();
	}
	if(RectCircleColliding(secondball, se10)){
		alert("YOU LOST");
	
		return count
		document.location.reload();
	}
	if(RectCircleColliding(secondball, se11)){
	
		return count
		document.location.reload();
	}
	if(RectCircleColliding(secondball, se12)){
		alert("YOU LOST");
		
		return count
		document.location.reload();
	}

	
	if(RectCircleColliding(secondball, seGoal)){
		 count=count+1
		
	 console.log("bhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh",count)
	   	return count
	}
}


function start(){
	//document.location.reload();
	requestAnimationFrame(start);
		
	c.clearRect(0, 0, innerWidth, innerHeight);
	var context = canvas.getContext("2d");
    var bw = 702;
    var bh = 386;
    var p = 7;
    var cw = bw + (p*2) + 2;
    var ch = bh + (p*2) + 2;

  
   
	// for 1st stage ....
	// Draw backgroung image

    function make_base()
	  {
	  base_image = new Image();
	  base_image.src = '/static/images/background.jpg';
	  context.globalAlpha = 0.5;
      context.drawImage(base_image, 0, 0,745,420);
	  context.globalAlpha = 1.0;
	  }

	   // other images
	  function make_child()
	  {
	  base_image = new Image();
	  base_image.src = '/static/images/child.png';
	  // context.globalAlpha = 0.5;
      context.drawImage(base_image, 120, 130,70,40);
	  // context.globalAlpha = 1.0;
	  // mage = new Image();
	  // mage.src ='static/images/stop.png';
	  // context.drawImage(mage, 520, 345,70,40);
	  
	  mage = new Image();
	  mage.src = 'static/images/mom.jpg';
	  context.drawImage(mage, 620, 75,72,38);
	  // chilfren image
	  child = new Image();
	  child.src = './static/images/children.jpg';
	  // context.globalAlpha = 0.5;
      context.drawImage(child, 420, 235,70,40);
	  // context.globalAlpha = 1.0;
	  
	  candey = new Image();
	  candey.src = '/static/images/candey.jpg';
	
      context.drawImage(candey, 636, 345,50,40);
	
	  }
	  // draw stop images
	  function make_image()
	  {
	  base_image = new Image();
	  base_image.src = '/static/images/stop.png';
	  context.globalAlpha = 0.5;
      context.drawImage(base_image, 120, 20,60,30);
	  context.globalAlpha = 1.0;
	  context.globalAlpha = 0.5;
      context.drawImage(base_image, 120, 75,60,30);
	  context.globalAlpha = 1.0;
	  context.globalAlpha = 0.5;
      context.drawImage(base_image, 120, 185,60,30);
	  context.globalAlpha = 1.0;
	  context.globalAlpha = 0.5;
      context.drawImage(base_image, 120, 235,60,30);
	  context.globalAlpha = 1.0;
	  context.globalAlpha = 0.5;
      context.drawImage(base_image, 120, 290,60,30);
	  context.globalAlpha = 1.0;
	   context.globalAlpha = 0.5;
      context.drawImage(base_image, 320, 20,60,30);
	  context.globalAlpha = 1.0;
	  context.globalAlpha = 0.5;
      context.drawImage(base_image, 320, 235,60,30);
	  context.globalAlpha = 1.0;
	  context.globalAlpha = 0.5;
      context.drawImage(base_image, 320, 345,60,30);
	  context.globalAlpha = 1.0;
	  context.globalAlpha = 0.5;
      context.drawImage(base_image, 625, 185,60,30);
	  context.globalAlpha = 1.0;
	  context.globalAlpha = 0.5;
      context.drawImage(base_image, 625, 290,60,30);
	  context.globalAlpha = 1.0;
	  }
	 function drawBoard(){
	 make_base();
	 make_image();
	 make_child();
        for (var x = 0; x < bw; x += 100) {
            context.moveTo(0.5 + x + p, p);
            context.lineTo(0.5 + x + p, bh + p);
        }

        for (var x = 0; x < bh; x += 55) {
            context.moveTo(p, 0.5 + x + p);
            context.lineTo(bw + p, 0.5 + x + p);
        }

        context.strokeStyle = "black";
        context.stroke();
    }
	  
	  // end for 1st stage
	  
	  // // for 2nd stagefunction make_base()
	
	  // function make_base_forSecondStage()
	  // {
	  // base_image = new Image();
	  // base_image.src = '/static/images/close_background.jpg';
	  // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 0, 0,745,420);
	  // context.globalAlpha = 1.0;
	  // }
	   // // other images
	  // function make_child_forSecondStage()
	  // {
	  // // base_image = new Image();
	  // // base_image.src = '/static/images/child.png';
	  // // // context.globalAlpha = 0.5;
      // // context.drawImage(base_image, 120, 130,70,40);
	  // // // context.globalAlpha = 1.0;
	  // // mage = new Image();
	  // // mage.src ='static/images/stop.png';
	  // // context.drawImage(mage, 520, 345,70,40);
	  
	  // // mage = new Image();
	  // // mage.src = 'static/images/mom.jpg';
	  // // context.drawImage(mage, 620, 75,72,38);
	  // // // chilfren image
	  // // child = new Image();
	  // // child.src = './static/images/children.jpg';
	  // // // context.globalAlpha = 0.5;
      // // context.drawImage(child, 420, 235,70,40);
	  // // // context.globalAlpha = 1.0;
	  
	  // candey = new Image();
	  // candey.src = '/static/images/candey.jpg';
	
      // context.drawImage(candey, 620, 340,60,50);
	
	  // }
	  // // draw stop images
	  // function make_image_forSecondStage()
	  // {
	  // base_image = new Image();
	  // base_image.src = '/static/images/stop.png';
	  // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 150, 90,60,30);
	  // context.globalAlpha = 1.0;
	  // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 35, 220,60,30);
	  // context.globalAlpha = 1.0;
	  // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 270, 220,60,30);
	  // context.globalAlpha = 1.0;
	  // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 270, 155,60,30);
	  // context.globalAlpha = 1.0;
	  // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 270, 280,60,30);
	  // context.globalAlpha = 1.0;
	   // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 150, 280,60,30);
	  // context.globalAlpha = 1.0;
	  // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 500, 345,60,30);
	  // context.globalAlpha = 1.0;
	  // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 500, 220,60,30);
	  // context.globalAlpha = 1.0;
	  // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 500, 90,60,30);
	  // context.globalAlpha = 1.0;
	  // context.globalAlpha = 0.5;
	  // context.drawImage(base_image, 610, 90,60,30);
	  // context.globalAlpha = 1.0;
	  // context.globalAlpha = 0.5;
	  // context.drawImage(base_image, 610, 220,60,30);
	  // context.globalAlpha = 1.0;
	  // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 615, 25,60,30);
	  // context.globalAlpha = 1.0;
	  // }
	  
	 // // for 2nd stage
	// function drawBoard_forSecondStage(){
	 // make_child_forSecondStage();
	 // make_base_forSecondStage();
	 // make_image_forSecondStage()
	
        // for (var x = 0; x < bw; x += 116.5) {
            // context.moveTo(0.5 + x + p, p);
            // context.lineTo(0.5 + x + p, bh + p);
        // }

        // for (var x = 0; x < bh; x += 64) {
            // context.moveTo(p, 0.5 + x + p);
            // context.lineTo(bw + p, 0.5 + x + p);
        // }

        // context.strokeStyle = "black";
        // context.stroke();
    // }
	// end for 2nd stage........................................................................................................
	// 3rd stage 
	// background image 
	  // function make_base_forThirdStage()
	  // {
	  // base_image = new Image();
	  // base_image.src = '/static/images/hospital.jpg';
	  // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 0, 0,745,420);
	  // context.globalAlpha = 1.0;
	  // }
	   // // 2nd hospital image
	  // function make_child_forThirdStage()
	  // {
	  // base_image = new Image();
	  // base_image.src = '/static/images/hospitall.jpg';
      // context.drawImage(base_image, 600, 330,100,65);
    
	
	
	  // }
	  // // draw stop images, x difference =120 y=130
	  // function make_image_forThirdStage()
	  // {
	  // base_image = new Image();
	  // base_image.src = '/static/images/stop.png';
	  // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 30, 90,60,30);
	  // context.globalAlpha = 1.0;
	  // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 150, 90,60,30);
	  // context.globalAlpha = 1.0;
	  // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 380, 150,60,30);
	  // context.globalAlpha = 1.0;

	  // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 500, 150,60,30);
	  // context.globalAlpha = 1.0;
	  // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 500, 280,60,30);
	  // context.globalAlpha = 1.0;
	
	
	  // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 260, 280,60,30);
	  // context.globalAlpha = 1.0;
	  // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 380, 25,60,30);
	  // context.globalAlpha = 1.0;
	 
	  

  
	  // }
	  
	// function drawBoard_forThirdStage(){
	 // make_base_forThirdStage();
	 // make_child_forThirdStage();
	 // make_image_forThirdStage();
	
        // for (var x = 0; x < bw; x += 116.5) {
            // context.moveTo(0.5 + x + p, p);
            // context.lineTo(0.5 + x + p, bh + p);
        // }

        // for (var x = 0; x < bh; x += 64) {
            // context.moveTo(p, 0.5 + x + p);
            // context.lineTo(bw + p, 0.5 + x + p);
        // }

        // context.strokeStyle = "black";
        // context.stroke();
    // }
	// // end for 3rd stage .......................................................................................................
	
	// // For fourth stage 
	
	
	
	  // function make_base_forFourthStage()
	  // {
	  // base_image = new Image();
	  // base_image.src = '/static/images/bottles.jpg';
	  // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 0, 0,745,420);
	  // context.globalAlpha = 1.0;
	  // }
	   // // other images
	  // function make_child_forFourthStage()
	  // {
	  // base_image = new Image();
	  // base_image.src = '/static/images/bott.jpg';
	  // // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 610, 330,90,50);
	  
	
	  // }
	  // // draw stop images, x difference =120 y=130
	  // function make_image_forFourthStage()
	  // {
	  // base_image = new Image();
	  // base_image.src = '/static/images/stop.png';
	  // // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 30, 90,60,30);
	  // // context.globalAlpha = 1.0;
	  // // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 130, 90,60,30);
	  // // context.globalAlpha = 1.0;
	  // // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 230, 150,60,30);
	  // // context.globalAlpha = 1.0;

	  // // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 330, 210,60,30);
	  // // context.globalAlpha = 1.0;
	  // // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 430, 280,60,30);
	  // // context.globalAlpha = 1.0;
	  // // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 430, 150,60,30);
	  // // context.globalAlpha = 1.0;
	  // // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 330, 30,60,30);
	  // // context.globalAlpha = 1.0;
	  // // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 130, 280,60,30);
	  // // context.globalAlpha = 1.0;
	  
	  // // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 530, 90,60,30);
	  // // context.globalAlpha = 1.0;
	  // // context.globalAlpha = 0.5;
      // // context.drawImage(base_image, 130, 170,60,30);
	  // // context.globalAlpha = 1.0;

  
	  // }
	  
	// function drawBoard_forFourthStage(){
	 // make_base_forFourthStage();
	 // make_image_forFourthStage();
	 // make_child_forFourthStage();
	
        // for (var x = 0; x < bw; x += 100) {
            // context.moveTo(0.5 + x + p, p);
            // context.lineTo(0.5 + x + p, bh + p);
        // }

        // for (var x = 0; x < bh; x += 64) {
            // context.moveTo(p, 0.5 + x + p);
            // context.lineTo(bw + p, 0.5 + x + p);
        // }

        // context.strokeStyle = "black";
        // context.stroke();
    // }
	// // end for fourth stage 
	
	  var stage=0;
	  // stage=0;
	  // for (var x = 0; x < 3; x += 1){
		  // if(stage ==0){
			  // drawBoard();
			  // ball.draw();
	          // stage=goalCheck(stage);
		
	          // console.log("myalert",stage);
			
		  // }
		  // else if(stage==1){
			
			  // c.clearRect(0, 0, innerWidth, innerHeight);
			 
			  // drawBoard_forSecondStage();
			  // secondball.draw();
			
	           // stage=goalCheckForSecond(stage);
			   // stage_count=stage;
		       // console.log("2ndstafe",stage_count,stage);

		  // }
		  // else if(stage==2){
			  // c.clearRect(0, 0, innerWidth, innerHeight);
			  // drawBoard_forThirdStage();
			  // ball.draw();
	        
		    // console.log(stage_count);

		  // }
	  // }
	  
	
	 c.clearRect(0, 0, innerWidth, innerHeight);
	drawBoard();
    ball.draw();
    // stage_count=goalCheck(stage_count);
   

   // drawBoard_forSecondStage();
   // ball.draw();
   // secondball.Draw();
 //  stage_count=goalCheck(stage_count);


    // drawBoard_forThirdStage();
    // ball.draw();
	// stage=goalCheck(stage);

     // drawBoard_forFourthStage();
	 // ball.draw();


}
//...........................................................................................................................



function startsec(){
	//document.location.reload();
	requestAnimationFrame(startsec);
	var context = canvas.getContext("2d");	
	c.clearRect(0, 0, innerWidth, innerHeight);
    var bw = 702;
    var bh = 386;
    var p = 7;
    var cw = bw + (p*2) + 2;
    var ch = bh + (p*2) + 2;

  
   
	// for 1st stage ....
	// Draw backgroung image

    // function make_base()
	  // {
	  // base_image = new Image();
	  // base_image.src = '/static/images/background.jpg';
	  // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 0, 0,745,420);
	  // context.globalAlpha = 1.0;
	  // }

	   // // other images
	  // function make_child()
	  // {
	  // base_image = new Image();
	  // base_image.src = '/static/images/child.png';
	  // // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 120, 130,70,40);
	  // // context.globalAlpha = 1.0;
	  // // mage = new Image();
	  // // mage.src ='static/images/stop.png';
	  // // context.drawImage(mage, 520, 345,70,40);
	  
	  // mage = new Image();
	  // mage.src = 'static/images/mom.jpg';
	  // context.drawImage(mage, 620, 75,72,38);
	  // // chilfren image
	  // child = new Image();
	  // child.src = './static/images/children.jpg';
	  // // context.globalAlpha = 0.5;
      // context.drawImage(child, 420, 235,70,40);
	  // // context.globalAlpha = 1.0;
	  
	  // candey = new Image();
	  // candey.src = '/static/images/candey.jpg';
	
      // context.drawImage(candey, 636, 345,50,40);
	
	  // }
	  // // draw stop images
	  // function make_image()
	  // {
	  // base_image = new Image();
	  // base_image.src = '/static/images/stop.png';
	  // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 120, 20,60,30);
	  // context.globalAlpha = 1.0;
	  // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 120, 75,60,30);
	  // context.globalAlpha = 1.0;
	  // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 120, 185,60,30);
	  // context.globalAlpha = 1.0;
	  // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 120, 235,60,30);
	  // context.globalAlpha = 1.0;
	  // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 120, 290,60,30);
	  // context.globalAlpha = 1.0;
	   // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 320, 20,60,30);
	  // context.globalAlpha = 1.0;
	  // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 320, 235,60,30);
	  // context.globalAlpha = 1.0;
	  // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 320, 345,60,30);
	  // context.globalAlpha = 1.0;
	  // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 625, 185,60,30);
	  // context.globalAlpha = 1.0;
	  // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 625, 290,60,30);
	  // context.globalAlpha = 1.0;
	  // }
	 // function drawBoard(){
	 // make_base();
	 // make_image();
	 // make_child();
        // for (var x = 0; x < bw; x += 100) {
            // context.moveTo(0.5 + x + p, p);
            // context.lineTo(0.5 + x + p, bh + p);
        // }

        // for (var x = 0; x < bh; x += 55) {
            // context.moveTo(p, 0.5 + x + p);
            // context.lineTo(bw + p, 0.5 + x + p);
        // }

        // context.strokeStyle = "black";
        // context.stroke();
    // }
	  
	  // end for 1st stage
	  
	  // // for 2nd stagefunction make_base()
	
	  function make_base_forSecondStage()
	  {
	  base_image = new Image();
	  base_image.src = '/static/images/close_background.jpg';
	  context.globalAlpha = 0.5;
      context.drawImage(base_image, 0, 0,745,420);
	  context.globalAlpha = 1.0;
	  }
	   // other images
	  function make_child_forSecondStage()
	  {
	  // base_image = new Image();
	  // base_image.src = '/static/images/child.png';
	  // // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 120, 130,70,40);
	  // // context.globalAlpha = 1.0;
	  // mage = new Image();
	  // mage.src ='static/images/stop.png';
	  // context.drawImage(mage, 520, 345,70,40);
	  
	  // mage = new Image();
	  // mage.src = 'static/images/mom.jpg';
	  // context.drawImage(mage, 620, 75,72,38);
	  // // chilfren image
	  // child = new Image();
	  // child.src = './static/images/children.jpg';
	  // // context.globalAlpha = 0.5;
      // context.drawImage(child, 420, 235,70,40);
	  // // context.globalAlpha = 1.0;
	  
	  candey = new Image();
	  candey.src = '/static/images/candey.jpg';
	
      context.drawImage(candey, 620, 340,60,50);
	
	  }
	  // draw stop images
	  function make_image_forSecondStage()
	  {
	  base_image = new Image();
	  base_image.src = '/static/images/stop.png';
	  context.globalAlpha = 0.5;
      context.drawImage(base_image, 150, 90,60,30);
	  context.globalAlpha = 1.0;
	  context.globalAlpha = 0.5;
      context.drawImage(base_image, 35, 220,60,30);
	  context.globalAlpha = 1.0;
	  context.globalAlpha = 0.5;
      context.drawImage(base_image, 270, 220,60,30);
	  context.globalAlpha = 1.0;
	  context.globalAlpha = 0.5;
      context.drawImage(base_image, 270, 155,60,30);
	  context.globalAlpha = 1.0;
	  context.globalAlpha = 0.5;
      context.drawImage(base_image, 270, 280,60,30);
	  context.globalAlpha = 1.0;
	   context.globalAlpha = 0.5;
      context.drawImage(base_image, 150, 280,60,30);
	  context.globalAlpha = 1.0;
	  context.globalAlpha = 0.5;
      context.drawImage(base_image, 500, 345,60,30);
	  context.globalAlpha = 1.0;
	  context.globalAlpha = 0.5;
      context.drawImage(base_image, 500, 220,60,30);
	  context.globalAlpha = 1.0;
	  context.globalAlpha = 0.5;
      context.drawImage(base_image, 500, 90,60,30);
	  context.globalAlpha = 1.0;
	  context.globalAlpha = 0.5;
	  context.drawImage(base_image, 610, 90,60,30);
	  context.globalAlpha = 1.0;
	  context.globalAlpha = 0.5;
	  context.drawImage(base_image, 610, 220,60,30);
	  context.globalAlpha = 1.0;
	  context.globalAlpha = 0.5;
      context.drawImage(base_image, 615, 25,60,30);
	  context.globalAlpha = 1.0;
	  }
	  
	 // for 2nd stage
	function drawBoard_forSecondStage(){
	 make_child_forSecondStage();
	 make_base_forSecondStage();
	 make_image_forSecondStage()
	
        for (var x = 0; x < bw; x += 116.5) {
            context.moveTo(0.5 + x + p, p);
            context.lineTo(0.5 + x + p, bh + p);
        }

        for (var x = 0; x < bh; x += 64) {
            context.moveTo(p, 0.5 + x + p);
            context.lineTo(bw + p, 0.5 + x + p);
        }

        context.strokeStyle = "black";
        context.stroke();
    }
	// end for 2nd stage........................................................................................................
	// 3rd stage 
	// background image 
	  // function make_base_forThirdStage()
	  // {
	  // base_image = new Image();
	  // base_image.src = '/static/images/hospital.jpg';
	  // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 0, 0,745,420);
	  // context.globalAlpha = 1.0;
	  // }
	   // // 2nd hospital image
	  // function make_child_forThirdStage()
	  // {
	  // base_image = new Image();
	  // base_image.src = '/static/images/hospitall.jpg';
      // context.drawImage(base_image, 600, 330,100,65);
    
	
	
	  // }
	  // // draw stop images, x difference =120 y=130
	  // function make_image_forThirdStage()
	  // {
	  // base_image = new Image();
	  // base_image.src = '/static/images/stop.png';
	  // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 30, 90,60,30);
	  // context.globalAlpha = 1.0;
	  // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 150, 90,60,30);
	  // context.globalAlpha = 1.0;
	  // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 380, 150,60,30);
	  // context.globalAlpha = 1.0;

	  // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 500, 150,60,30);
	  // context.globalAlpha = 1.0;
	  // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 500, 280,60,30);
	  // context.globalAlpha = 1.0;
	
	
	  // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 260, 280,60,30);
	  // context.globalAlpha = 1.0;
	  // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 380, 25,60,30);
	  // context.globalAlpha = 1.0;
	 
	  

  
	  // }
	  
	// function drawBoard_forThirdStage(){
	 // make_base_forThirdStage();
	 // make_child_forThirdStage();
	 // make_image_forThirdStage();
	
        // for (var x = 0; x < bw; x += 116.5) {
            // context.moveTo(0.5 + x + p, p);
            // context.lineTo(0.5 + x + p, bh + p);
        // }

        // for (var x = 0; x < bh; x += 64) {
            // context.moveTo(p, 0.5 + x + p);
            // context.lineTo(bw + p, 0.5 + x + p);
        // }

        // context.strokeStyle = "black";
        // context.stroke();
    // }
	// // end for 3rd stage .......................................................................................................
	
	// // For fourth stage 
	
	
	
	  // function make_base_forFourthStage()
	  // {
	  // base_image = new Image();
	  // base_image.src = '/static/images/bottles.jpg';
	  // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 0, 0,745,420);
	  // context.globalAlpha = 1.0;
	  // }
	   // // other images
	  // function make_child_forFourthStage()
	  // {
	  // base_image = new Image();
	  // base_image.src = '/static/images/bott.jpg';
	  // // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 610, 330,90,50);
	  
	
	  // }
	  // // draw stop images, x difference =120 y=130
	  // function make_image_forFourthStage()
	  // {
	  // base_image = new Image();
	  // base_image.src = '/static/images/stop.png';
	  // // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 30, 90,60,30);
	  // // context.globalAlpha = 1.0;
	  // // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 130, 90,60,30);
	  // // context.globalAlpha = 1.0;
	  // // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 230, 150,60,30);
	  // // context.globalAlpha = 1.0;

	  // // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 330, 210,60,30);
	  // // context.globalAlpha = 1.0;
	  // // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 430, 280,60,30);
	  // // context.globalAlpha = 1.0;
	  // // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 430, 150,60,30);
	  // // context.globalAlpha = 1.0;
	  // // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 330, 30,60,30);
	  // // context.globalAlpha = 1.0;
	  // // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 130, 280,60,30);
	  // // context.globalAlpha = 1.0;
	  
	  // // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 530, 90,60,30);
	  // // context.globalAlpha = 1.0;
	  // // context.globalAlpha = 0.5;
      // // context.drawImage(base_image, 130, 170,60,30);
	  // // context.globalAlpha = 1.0;

  
	  // }
	  
	// function drawBoard_forFourthStage(){
	 // make_base_forFourthStage();
	 // make_image_forFourthStage();
	 // make_child_forFourthStage();
	
        // for (var x = 0; x < bw; x += 100) {
            // context.moveTo(0.5 + x + p, p);
            // context.lineTo(0.5 + x + p, bh + p);
        // }

        // for (var x = 0; x < bh; x += 64) {
            // context.moveTo(p, 0.5 + x + p);
            // context.lineTo(bw + p, 0.5 + x + p);
        // }

        // context.strokeStyle = "black";
        // context.stroke();
    // }
	// // end for fourth stage 
	
	  var stage=0;
	  // stage=0;
	  // for (var x = 0; x < 3; x += 1){
		  // if(stage ==0){
			  // drawBoard();
			  // ball.draw();
	          // stage=goalCheck(stage);
		
	          // console.log("myalert",stage);
			
		  // }
		  // else if(stage==1){
			
			  // c.clearRect(0, 0, innerWidth, innerHeight);
			 
			  // drawBoard_forSecondStage();
			  // secondball.draw();
			
	           // stage=goalCheckForSecond(stage);
			   // stage_count=stage;
		       // console.log("2ndstafe",stage_count,stage);

		  // }
		  // else if(stage==2){
			  // c.clearRect(0, 0, innerWidth, innerHeight);
			  // drawBoard_forThirdStage();
			  // ball.draw();
	        
		    // console.log(stage_count);

		  // }
	  // }
	  
	
	
	// drawBoard();
    // ball.draw();
    // stage_count=goalCheck(stage_count);
   
 c.clearRect(0, 0, innerWidth, innerHeight);
   drawBoard_forSecondStage();
   secondball.draw();
 


}


//........................................................................................startsec end.............................

function startthird(){
	//document.location.reload();
	requestAnimationFrame(startthird);
	var context = canvas.getContext("2d");	
	c.clearRect(0, 0, innerWidth, innerHeight);
    var bw = 702;
    var bh = 386;
    var p = 7;
    var cw = bw + (p*2) + 2;
    var ch = bh + (p*2) + 2;

  
   
	// end for 2nd stage........................................................................................................
	// 3rd stage 
	// background image 
	  function make_base_forThirdStage()
	  {
	  base_image = new Image();
	  base_image.src = '/static/images/hospital.jpg';
	  context.globalAlpha = 0.5;
      context.drawImage(base_image, 0, 0,745,420);
	  context.globalAlpha = 1.0;
	  }
	   // 2nd hospital image
	  function make_child_forThirdStage()
	  {
	  base_image = new Image();
	  base_image.src = '/static/images/hospitall.jpg';
      context.drawImage(base_image, 600, 330,100,65);
    
	
	
	  }
	  // draw stop images, x difference =120 y=130
	  function make_image_forThirdStage()
	  {
	  base_image = new Image();
	  base_image.src = '/static/images/stop.png';
	  context.globalAlpha = 0.5;
      context.drawImage(base_image, 30, 90,60,30);
	  context.globalAlpha = 1.0;
	  context.globalAlpha = 0.5;
      context.drawImage(base_image, 150, 90,60,30);
	  context.globalAlpha = 1.0;
	  context.globalAlpha = 0.5;
      context.drawImage(base_image, 380, 150,60,30);
	  context.globalAlpha = 1.0;

	  context.globalAlpha = 0.5;
      context.drawImage(base_image, 500, 150,60,30);
	  context.globalAlpha = 1.0;
	  context.globalAlpha = 0.5;
      context.drawImage(base_image, 500, 280,60,30);
	  context.globalAlpha = 1.0;
	
	
	  context.globalAlpha = 0.5;
      context.drawImage(base_image, 260, 280,60,30);
	  context.globalAlpha = 1.0;
	  context.globalAlpha = 0.5;
      context.drawImage(base_image, 380, 25,60,30);
	  context.globalAlpha = 1.0;
	 
	  

  
	  }
	  
	function drawBoard_forThirdStage(){
	 make_base_forThirdStage();
	 make_child_forThirdStage();
	 make_image_forThirdStage();
	
        for (var x = 0; x < bw; x += 116.5) {
            context.moveTo(0.5 + x + p, p);
            context.lineTo(0.5 + x + p, bh + p);
        }

        for (var x = 0; x < bh; x += 64) {
            context.moveTo(p, 0.5 + x + p);
            context.lineTo(bw + p, 0.5 + x + p);
        }

        context.strokeStyle = "black";
        context.stroke();
    }
	// // end for 3rd stage .......................................................................................................
	
	// // For fourth stage 
	
	
	
	  // function make_base_forFourthStage()
	  // {
	  // base_image = new Image();
	  // base_image.src = '/static/images/bottles.jpg';
	  // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 0, 0,745,420);
	  // context.globalAlpha = 1.0;
	  // }
	   // // other images
	  // function make_child_forFourthStage()
	  // {
	  // base_image = new Image();
	  // base_image.src = '/static/images/bott.jpg';
	  // // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 610, 330,90,50);
	  
	
	  // }
	  // // draw stop images, x difference =120 y=130
	  // function make_image_forFourthStage()
	  // {
	  // base_image = new Image();
	  // base_image.src = '/static/images/stop.png';
	  // // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 30, 90,60,30);
	  // // context.globalAlpha = 1.0;
	  // // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 130, 90,60,30);
	  // // context.globalAlpha = 1.0;
	  // // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 230, 150,60,30);
	  // // context.globalAlpha = 1.0;

	  // // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 330, 210,60,30);
	  // // context.globalAlpha = 1.0;
	  // // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 430, 280,60,30);
	  // // context.globalAlpha = 1.0;
	  // // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 430, 150,60,30);
	  // // context.globalAlpha = 1.0;
	  // // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 330, 30,60,30);
	  // // context.globalAlpha = 1.0;
	  // // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 130, 280,60,30);
	  // // context.globalAlpha = 1.0;
	  
	  // // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 530, 90,60,30);
	  // // context.globalAlpha = 1.0;
	  // // context.globalAlpha = 0.5;
      // // context.drawImage(base_image, 130, 170,60,30);
	  // // context.globalAlpha = 1.0;

  
	  // }
	  
	// function drawBoard_forFourthStage(){
	 // make_base_forFourthStage();
	 // make_image_forFourthStage();
	 // make_child_forFourthStage();
	
        // for (var x = 0; x < bw; x += 100) {
            // context.moveTo(0.5 + x + p, p);
            // context.lineTo(0.5 + x + p, bh + p);
        // }

        // for (var x = 0; x < bh; x += 64) {
            // context.moveTo(p, 0.5 + x + p);
            // context.lineTo(bw + p, 0.5 + x + p);
        // }

        // context.strokeStyle = "black";
        // context.stroke();
    // }
	// // end for fourth stage 
	
	  var stage=0;
	  // stage=0;
	  // for (var x = 0; x < 3; x += 1){
		  // if(stage ==0){
			  // drawBoard();
			  // ball.draw();
	          // stage=goalCheck(stage);
		
	          // console.log("myalert",stage);
			
		  // }
		  // else if(stage==1){
			
			  // c.clearRect(0, 0, innerWidth, innerHeight);
			 
			  // drawBoard_forSecondStage();
			  // secondball.draw();
			
	           // stage=goalCheckForSecond(stage);
			   // stage_count=stage;
		       // console.log("2ndstafe",stage_count,stage);

		  // }
		  // else if(stage==2){
			  // c.clearRect(0, 0, innerWidth, innerHeight);
			  // drawBoard_forThirdStage();
			  // ball.draw();
	        
		    // console.log(stage_count);

		  // }
	  // }
	  
	
	
	// drawBoard();
    // ball.draw();
    // stage_count=goalCheck(stage_count);
   
 c.clearRect(0, 0, innerWidth, innerHeight);
 
    drawBoard_forThirdStage();
    ball.draw();
	// stage=goalCheck(stage);

     // drawBoard_forFourthStage();
	 // ball.draw();


}

//.........................................................................start third end...........................




function startfourth(){
	//document.location.reload();
	requestAnimationFrame(startfourth);
	var context = canvas.getContext("2d");	
	c.clearRect(0, 0, innerWidth, innerHeight);
    var bw = 702;
    var bh = 386;
    var p = 7;
    var cw = bw + (p*2) + 2;
    var ch = bh + (p*2) + 2;

	
	// // For fourth stage 
	
	
	
	  function make_base_forFourthStage()
	  {
	  base_image = new Image();
	  base_image.src = '/static/images/bottles.jpg';
	  context.globalAlpha = 0.5;
      context.drawImage(base_image, 0, 0,745,420);
	  context.globalAlpha = 1.0;
	  }
	   // other images
	  function make_child_forFourthStage()
	  {
	  base_image = new Image();
	  base_image.src = '/static/images/bott.jpg';
	  // context.globalAlpha = 0.5;
      context.drawImage(base_image, 610, 330,90,50);
	  
	
	  }
	  // draw stop images, x difference =120 y=130
	  function make_image_forFourthStage()
	  {
	  base_image = new Image();
	  base_image.src = '/static/images/stop.png';
	  // context.globalAlpha = 0.5;
      context.drawImage(base_image, 30, 90,60,30);
	  // context.globalAlpha = 1.0;
	  // context.globalAlpha = 0.5;
      context.drawImage(base_image, 130, 90,60,30);
	  // context.globalAlpha = 1.0;
	  // context.globalAlpha = 0.5;
      context.drawImage(base_image, 230, 150,60,30);
	  // context.globalAlpha = 1.0;

	  // context.globalAlpha = 0.5;
      context.drawImage(base_image, 330, 210,60,30);
	  // context.globalAlpha = 1.0;
	  // context.globalAlpha = 0.5;
      context.drawImage(base_image, 430, 280,60,30);
	  // context.globalAlpha = 1.0;
	  // context.globalAlpha = 0.5;
      context.drawImage(base_image, 430, 150,60,30);
	  // context.globalAlpha = 1.0;
	  // context.globalAlpha = 0.5;
      context.drawImage(base_image, 330, 30,60,30);
	  // context.globalAlpha = 1.0;
	  // context.globalAlpha = 0.5;
      context.drawImage(base_image, 130, 280,60,30);
	  // context.globalAlpha = 1.0;
	  
	  // context.globalAlpha = 0.5;
      context.drawImage(base_image, 530, 90,60,30);
	  // context.globalAlpha = 1.0;
	  // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 130, 170,60,30);
	  // context.globalAlpha = 1.0;

  
	  }
	  
	function drawBoard_forFourthStage(){
	 make_base_forFourthStage();
	 make_image_forFourthStage();
	 make_child_forFourthStage();
	
        for (var x = 0; x < bw; x += 100) {
            context.moveTo(0.5 + x + p, p);
            context.lineTo(0.5 + x + p, bh + p);
        }

        for (var x = 0; x < bh; x += 64) {
            context.moveTo(p, 0.5 + x + p);
            context.lineTo(bw + p, 0.5 + x + p);
        }

        context.strokeStyle = "black";
        context.stroke();
    }
	// // end for fourth stage 
	
	  var stage=0;
	  
	  
	
	
	// drawBoard();
    // ball.draw();
    // stage_count=goalCheck(stage_count);
   
 c.clearRect(0, 0, innerWidth, innerHeight);
  


     drawBoard_forFourthStage();
	 ball.draw();


}





function stagesecondcall(){
	requestAnimationFrame(start);
	var context = canvas.getContext("2d");
	var bw = 702;
    var bh = 386;
    var p = 7;
    var cw = bw + (p*2) + 2;
    var ch = bh + (p*2) + 2;
  
   // var context = canvas.getContext("2d");
	
	 // for 2nd stagefunction make_base()
	
	  function make_base_forSecondStage()
	  {
	  base_image = new Image();
	  base_image.src = '/static/images/close_background.jpg';
	  context.globalAlpha = 0.5;
      context.drawImage(base_image, 0, 0,745,420);
	  context.globalAlpha = 1.0;
	  }
	   // other images
	  function make_child_forSecondStage()
	  {
	  // base_image = new Image();
	  // base_image.src = '/static/images/child.png';
	  // // context.globalAlpha = 0.5;
      // context.drawImage(base_image, 120, 130,70,40);
	  // // context.globalAlpha = 1.0;
	  // mage = new Image();
	  // mage.src ='static/images/stop.png';
	  // context.drawImage(mage, 520, 345,70,40);
	  
	  // mage = new Image();
	  // mage.src = 'static/images/mom.jpg';
	  // context.drawImage(mage, 620, 75,72,38);
	  // // chilfren image
	  // child = new Image();
	  // child.src = './static/images/children.jpg';
	  // // context.globalAlpha = 0.5;
      // context.drawImage(child, 420, 235,70,40);
	  // // context.globalAlpha = 1.0;
	  
	  candey = new Image();
	  candey.src = '/static/images/candey.jpg';
	
      context.drawImage(candey, 620, 340,60,50);
	
	  }
	  // draw stop images
	  function make_image_forSecondStage()
	  {
	  base_image = new Image();
	  base_image.src = '/static/images/stop.png';
	  context.globalAlpha = 0.5;
      context.drawImage(base_image, 150, 90,60,30);
	  context.globalAlpha = 1.0;
	  context.globalAlpha = 0.5;
      context.drawImage(base_image, 35, 220,60,30);
	  context.globalAlpha = 1.0;
	  context.globalAlpha = 0.5;
      context.drawImage(base_image, 270, 220,60,30);
	  context.globalAlpha = 1.0;
	  context.globalAlpha = 0.5;
      context.drawImage(base_image, 270, 155,60,30);
	  context.globalAlpha = 1.0;
	  context.globalAlpha = 0.5;
      context.drawImage(base_image, 270, 280,60,30);
	  context.globalAlpha = 1.0;
	   context.globalAlpha = 0.5;
      context.drawImage(base_image, 150, 280,60,30);
	  context.globalAlpha = 1.0;
	  context.globalAlpha = 0.5;
      context.drawImage(base_image, 500, 345,60,30);
	  context.globalAlpha = 1.0;
	  context.globalAlpha = 0.5;
      context.drawImage(base_image, 500, 220,60,30);
	  context.globalAlpha = 1.0;
	  context.globalAlpha = 0.5;
      context.drawImage(base_image, 500, 90,60,30);
	  context.globalAlpha = 1.0;
	  context.globalAlpha = 0.5;
	  context.drawImage(base_image, 610, 90,60,30);
	  context.globalAlpha = 1.0;
	  context.globalAlpha = 0.5;
	  context.drawImage(base_image, 610, 220,60,30);
	  context.globalAlpha = 1.0;
	  context.globalAlpha = 0.5;
      context.drawImage(base_image, 615, 25,60,30);
	  context.globalAlpha = 1.0;
	  }
	  
	 // for 2nd stage
	function drawBoard_forSecondStage(){
	 make_child_forSecondStage();
	 make_base_forSecondStage();
	 make_image_forSecondStage()
	
        for (var x = 0; x < bw; x += 116.5) {
            context.moveTo(0.5 + x + p, p);
            context.lineTo(0.5 + x + p, bh + p);
        }

        for (var x = 0; x < bh; x += 64) {
            context.moveTo(p, 0.5 + x + p);
            context.lineTo(bw + p, 0.5 + x + p);
        }

        context.strokeStyle = "black";
        context.stroke();
    }
	
	c.clearRect(0, 0, innerWidth, innerHeight);
			 
    drawBoard_forSecondStage();
	secondball.draw();
	console.log("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb");
	
}

// Event that check if keys are pressed
document.addEventListener('keydown', (e) => {
	if(e.keyCode === 37){
		keyPressed.left = true;
	}
	if(e.keyCode === 39){
		keyPressed.right = true;
	}
	if(e.keyCode === 38){
		keyPressed.up = true;
	}
	if(e.keyCode === 40){
		keyPressed.down = true;
	}
})

// Event that check if keys aren't pressed
document.addEventListener('keyup', (e) => {
	if(e.keyCode === 37){
		keyPressed.left = false;
	}
	if(e.keyCode === 39){
		keyPressed.right = false;
	}
	if(e.keyCode === 38){
		keyPressed.up = false;
	}
	if(e.keyCode === 40){
		keyPressed.down = false;
	}
});

//start();



/*
// Check function if you touch walls
function checkTouch(){

	for(let i = 0; i < wallArray.lenght; i++){
		let distX = Math.abs(ball.x - wallArray[i].x - wallArray[i].width / 2);
	    let distY = Math.abs(ball.y - wallArray[i].y - wallArray[i].height / 2);

		if (distX > (wallArray[i].width / 2 + ball.radius)) { return false; };
    	if (distY > (wallArray[i].height / 2 + ball.radius)) { return false; };

    	if (distX <= (wallArray[i].width / 2)) { return true; };
    	if (distY <= (wallArray[i].height / 2)) { return true; };

    	let dx = distX - wallArray[i].width / 2;
    	let dy = distY - wallArray[i].height / 2;

    	return (dx*dx+dy*dy<=(ball.radius * ball.radius));
	}
    
}*/
