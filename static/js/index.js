import * as THREE from 'https://threejsfundamentals.org/threejs/resources/threejs/r132/build/three.module.js';
import {OrbitControls} from 'https://threejsfundamentals.org/threejs/resources/threejs/r132/examples/jsm/controls/OrbitControls.js';
import {GLTFLoader} from 'https://threejsfundamentals.org/threejs/resources/threejs/r132/examples/jsm/loaders/GLTFLoader.js';
import Stats from 'https://threejsfundamentals.org/threejs/resources/threejs/r132/examples/jsm/libs/stats.module.js'

var container, stats, controls, mixer, clock;
var camera, scene, renderer;

var moveForward = false, moveBackward = false, turnLeft = false, turnRight = false;
var model, lastAction, activeAction, current_walkSpeed = 0, current_turnSpeed = 0;

var mouse = new THREE.Vector2(), raycaster = new THREE.Raycaster(), objects = [];
var chatbot;

init();
animate();
onWindowResize();

function init() {
	// Create Scene
	container = document.createElement( 'div' );
	document.body.appendChild( container );
	// Set Camera
	camera = new THREE.PerspectiveCamera( 75, window.innerWidth / (window.innerHeight * 0.9), 0.5, 500 );
	camera.position.set( -30, 20, -30 );
	// Set Scene
	scene = new THREE.Scene();
	scene.background = new THREE.Color( 0x6dddff );
	
	clock = new THREE.Clock();

	// Light From Camera
	const hemiLight = new THREE.HemisphereLight( 0xffffff, 0x444444 );
	hemiLight.position.set( 0, 200, 0 );
	scene.add( hemiLight );
	// Light of Sun
	const dirLight = new THREE.DirectionalLight( 0xffffff );
	dirLight.position.set( 0, 200, 100 );
	dirLight.castShadow = true;
	dirLight.shadow.camera.top = 180;
	dirLight.shadow.camera.bottom = - 100;
	dirLight.shadow.camera.left = - 120;
	dirLight.shadow.camera.right = 120;
	scene.add( dirLight );

	// Load Models
	loadModel();

	// Set Renderer
	renderer = new THREE.WebGLRenderer( { antialias: true, alpha: true } );
	renderer.setSize( window.innerWidth, window.innerHeight * 0.9 );
	renderer.setPixelRatio( Math.min(window.devicePixelRatio, 2) );
	renderer.setAnimationLoop();
	renderer.gammaOutput = true;
	container.appendChild( renderer.domElement );

	// Event Listener
	window.addEventListener( 'resize', onWindowResize, true );
	document.addEventListener( 'mousedown', onDocumentMouseDown, false );

	// Create controler
	controls = new OrbitControls( camera, renderer.domElement );
	controls.enableZoom = true
	controls.enableDamping = true
	controls.minDistance = 10;
	controls.maxDistance = 40;
	controls.target.set( -48, 5, -48 );
	controls.update();

	// Stats
	stats = new Stats();
	container.appendChild( stats.dom );

	// Make mouse control smooth
	const tick = ()=>{
		renderer.render(scene, camera)
		window.requestAnimationFrame(tick)
		controls.update()
	}
	tick()

	// Action when button clicked
	/*const btn1 = document.querySelector('#button1');
	btn1.addEventListener('click', () => { // Send EEG
		camera.position.set(-30, 50, -60);
		window.open("http://" + location.hostname + ":" + location.port + "/eeg.html");
	})
	const btn3 = document.querySelector('#button2');
	btn3.addEventListener('click', () => { // Hospital
		camera.position.set(-60, 50, -30);
		window.open("http://" + location.hostname + ":" + location.port + "/map.html");
	})
	const btn4 = document.querySelector('#button3');
	btn4.addEventListener('click', () => { // Record
		camera.position.set(-30, 50, -30);
		window.open("http://" + location.hostname + ":" + location.port + "/graph.html");
	})*/
}

function animate() {
	requestAnimationFrame( animate );
	var delta = clock.getDelta();
	if ( mixer ) mixer.update( delta );
	if ( model ) moveModel(); // KeyDown & KeyUp
	renderer.render( scene, camera );
	stats.update();
}

function onWindowResize() {
	camera.aspect = window.innerWidth / (window.innerHeight * 0.9);
	camera.updateProjectionMatrix();
	renderer.setSize( window.innerWidth, window.innerHeight * 0.9 );
}

function loadModel() {
	// GLTF Object with animation - Model
	var loader = new GLTFLoader().setPath('./static/assets/')
	loader.load('merged_y.glb', function(gltf) {
		gltf.scene.scale.setScalar(0.13)
		gltf.scene.position.set(-47, 0, -47)
		scene.add( gltf.scene );

		mixer = new THREE.AnimationMixer( gltf.scene );
		var action0 = mixer.clipAction( gltf.animations[ 0 ] ); // Walking
		var action1 = mixer.clipAction( gltf.animations[ 1 ] ); // Standing
		action0.play();
		action0.setEffectiveWeight(0);
		action1.play();
		lastAction = action0;
		activeAction = action1;
		
		model = gltf.scene
		// Move Character with Key
		// 87-w / 83-s / 65-a / 68-d / 32 - space
		document.addEventListener("keydown", onDocumentKeyDown, false);
		function onDocumentKeyDown(event) {
			var keyCode = event.which;
			if (keyCode == 87) {
				moveForward = true
                setAction(action0);
			} else if (keyCode == 83) {
				moveBackward = true
                setAction(action0);
			} else if (keyCode == 65) {
				turnLeft = true
                setAction(action0);
			} else if (keyCode == 68) {
				turnRight = true
                setAction(action0);
			}
		};
		document.addEventListener("keyup", onDocumentKeyUp, false);
		function onDocumentKeyUp(event) {
			var keyCode = event.which;
			if (keyCode == 87) {
				moveForward = false
                setAction(action1);
			} else if (keyCode == 83) {
				moveBackward = false
                setAction(action1);
			} else if (keyCode == 65) {
				turnLeft = false
                setAction(action1);
			} else if (keyCode == 68) {
				turnRight = false
                setAction(action1);
			}
		};
		objects.push(model); // clickable model
	})
	chatbot = true;

	// GLB Object without animation - City
	loader.load('city.glb', function(gltf){
		gltf.scene.scale.setScalar(0.1)
		gltf.scene.position.set(0, 12, 0)
		scene.add(gltf.scene);
	})
}

function moveModel() {
	// smooth animation
	if(lastAction.getEffectiveWeight() > 0)
		lastAction.setEffectiveWeight(lastAction.getEffectiveWeight() - 0.05)
	if(activeAction.getEffectiveWeight() < 1)
		activeAction.setEffectiveWeight(activeAction.getEffectiveWeight() + 0.05)

	// smooth movement
	var walkSpeed = 0.3
	var turnSpeed = Math.PI/45;
	if(moveForward || moveBackward ) {
		if(current_walkSpeed < walkSpeed) {
			current_walkSpeed += walkSpeed * 0.01
		}
	} else {
		if(current_walkSpeed > 0) {
			current_walkSpeed -= walkSpeed * 0.01
		}
	}
	if(turnLeft || turnRight ) {
		if(current_turnSpeed < turnSpeed) {
			current_turnSpeed += turnSpeed * 0.01
		}
	} else {
		if(current_turnSpeed > 0) {
			current_turnSpeed -= turnSpeed * 0.01
		}
	}
	
	// movement speed
	if(moveForward) {
		var direction = new THREE.Vector3()
		model.getWorldDirection(direction);
		direction.multiplyScalar(current_walkSpeed)
		model.position.add(direction);
	}
	if(moveBackward) {
		var direction = new THREE.Vector3()
		model.getWorldDirection(direction);
		
		direction.multiplyScalar(current_walkSpeed)
		direction.negate()
		model.position.add(direction);	
	}
	if(turnLeft) {
		model.rotation.y += current_turnSpeed;
	}
	if(turnRight) {
		model.rotation.y -= current_turnSpeed;
	}

	// Up-Right
	if(model.position.x > -30 && model.position.z < -60) {
		keyRest()
		var result = confirm("페이지가 전환됩니다.")
		if(result) {
			window.open("https://www.youtube.com/", "page")
		}
	}
	// Up-Left
	if(model.position.x < -60 && model.position.z < -60) {
		keyRest()
		var result = confirm("페이지가 전환됩니다.")
		if(result) {
			window.open("https://www.youtube.com/", "page")
		}
	}
	// Down-Left
	if(model.position.x < -60 && model.position.z > -30) {
		keyRest()
		var result = confirm("페이지가 전환됩니다.")
		if(result) {
			window.open("https://www.youtube.com/", "page")
		}
	}
	// Down-Right
	if(model.position.x > -30 && model.position.z > -30) {
		keyRest()
		var result = confirm("페이지가 전환됩니다.")
		if(result) {
			window.open("https://www.youtube.com/", "page")
		}
	}
}

function keyRest() {
	moveForward = false
	moveBackward = false
	turnLeft = false
	turnRight = false
	model.position.set(-47, 0, -47)
}

function setAction (toAction) {	
	if (toAction == activeAction) {		
		return
	} else {
		lastAction = activeAction
		activeAction = toAction
		
		lastAction.stop()
		lastAction.setEffectiveWeight(lastAction.getEffectiveWeight() - 0.05)
		activeAction.reset()
		activeAction.setEffectiveWeight(activeAction.getEffectiveWeight() + 0.05)
		activeAction.play()
	}
}

// Recognize Character Click
function onDocumentMouseDown(event) {
	event.preventDefault();

	mouse.x = ( event.clientX / window.innerWidth ) * 2 - 1;
	mouse.y = - ( event.clientY / (window.innerHeight * 0.9) - (window.innerHeight * 0.0001)) * 2 + 1;

	raycaster.setFromCamera( mouse, camera );
	var intersections = raycaster.intersectObjects(objects, true);
	if ( intersections.length > 0 ) {
		const object = intersections[ 0 ].object;
		callChatBot();
	}
}

function callChatBot() {
	if(!chatbot){
		chatbot = true;
		var rasa = document.getElementById("rasaWebchatPro");
		rasa.style.display = "block";
	} else {
		chatbot = false;
		var rasa = document.getElementById("rasaWebchatPro");
		rasa.style.display = "none";
	}
}