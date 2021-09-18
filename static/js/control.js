import * as THREE from 'https://threejsfundamentals.org/threejs/resources/threejs/r132/build/three.module.js';
import {OrbitControls} from 'https://threejsfundamentals.org/threejs/resources/threejs/r132/examples/jsm/controls/OrbitControls.js';
import {GLTFLoader} from 'https://threejsfundamentals.org/threejs/resources/threejs/r132/examples/jsm/loaders/GLTFLoader.js';
import Stats from 'https://threejsfundamentals.org/threejs/resources/threejs/r132/examples/jsm/libs/stats.module.js'

import {FBXLoader} from 'https://threejsfundamentals.org/threejs/resources/threejs/r132/examples/jsm/loaders/FBXLoader.js'
import {DRACOLoader} from 'https://threejsfundamentals.org/threejs/resources/threejs/r132/examples/jsm/loaders/DRACOLoader.js'
import vShader from '../shaders/vertexShader.glsl.js'
import fShader from '../shaders/fragmentShader.glsl.js'

var container, stats, controls, mixer, clock;
var camera, scene, renderer;

var moveForward = false, moveBackward = false, turnLeft = false, turnRight = false;
var model, lastAction, activeAction, current_walkSpeed = 0, current_turnSpeed = 0;

init();
animate();
onWindowResize();

function init() {
	// Create Scene
	container = document.createElement( 'div' );
	document.body.appendChild( container );
	// Set Camera
	camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.5, 500 );
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
	keyRest();
	
	// Set Renderer
	renderer = new THREE.WebGLRenderer( { antialias: true, alpha: true } );
	renderer.setSize( window.innerWidth, window.innerHeight );
	renderer.setPixelRatio( Math.min(window.devicePixelRatio, 2) );
	renderer.setAnimationLoop()
	renderer.gammaOutput = true;
	container.appendChild( renderer.domElement );

	// Create controler
	controls = new OrbitControls( camera, renderer.domElement );
	controls.enableZoom = true
	controls.enableDamping = true
	controls.minDistance = 10;
	controls.maxDistance = 40;
	controls.target.set( -48, 5, -48 );
	controls.update();

	window.addEventListener( 'resize', onWindowResize, true );

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
	camera.aspect = window.innerWidth / window.innerHeight;
	camera.updateProjectionMatrix();
	renderer.setSize( window.innerWidth, window.innerHeight );
}

function loadModel() {
	// GLTF Object with animation - mesh
	var loader = new GLTFLoader().setPath('./assets/')
	loader.load('merged_y.glb', function(gltf) {
		gltf.scene.scale.setScalar(0.13)
		gltf.scene.position.set(-47, 0, -47)
		scene.add( gltf.scene );

		mixer = new THREE.AnimationMixer( gltf.scene );
		var action0 = mixer.clipAction( gltf.animations[ 0 ] ); // Walking
		var action1 = mixer.clipAction( gltf.animations[ 1 ] ); // Standing
		//action0.play();
		action1.play();
		lastAction = action1;
		activeAction = action1;
		// action0.setEffectiveWeight(0)
		// action1.setEffectiveWeight(1)

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
			} else if (keyCode == 32) {
				gltf.scene.position.set(-47, 0, -47)
				console.log("RESET POSITION BY SPACE BAR")
			}
			// action0.fadeOut(0.5)
			// action0.stop();
			// action1.fadeIn(0.5)
			// action1.play();
			//action0.crossFadeFrom(action1, 1, true);
			//action0.fadeOut(1)
			//action1.fadeIn(1)
			//action0.setEffectiveWeight(0)
			//action1.setEffectiveWeight(1)
			//action0.stop();
			//action1.play();
			//console.log("RESTART WALKING")
		};
	})

	// // GLTF Object without animation - Chair
	// loader.load('chair.gltf', function(gltf) {
	// 	gltf.scene.scale.set(5.0, 5.5, 5.0)
	// 	gltf.scene.position.set(-47, 0, -49)
	// 	scene.add( gltf.scene )
	// })

	loader.load('city.glb', function(gltf){
		gltf.scene.scale.setScalar(0.1)
		gltf.scene.position.set(0, 12, 0)
		scene.add(gltf.scene);
	})

	// TEST BOX
	const geometry = new THREE.BoxGeometry(5, 5, 5)
	const material = new THREE.MeshBasicMaterial({ color: 0xff0000 })
	const boxMesh = new THREE.Mesh(geometry, material)
	boxMesh.position.set(0, 0, 0)
	scene.add(boxMesh)
}

function moveModel() {
	// smooth movement
	var walkSpeed = 0.2
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
}

function keyRest() {
	moveForward = false
	moveBackward = false
	turnLeft = false
	turnRight = false
}

function setAction (toAction) {	
	if (toAction == activeAction) {
		return;
	} else {
		lastAction = activeAction
		activeAction = toAction
		//lastAction.stop()
		lastAction.fadeOut(1)
		activeAction.reset()
		activeAction.fadeIn(1)
		activeAction.play()
	}
}
