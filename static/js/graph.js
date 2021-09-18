import * as THREE from 'https://threejsfundamentals.org/threejs/resources/threejs/r132/build/three.module.js';
import {OrbitControls} from 'https://threejsfundamentals.org/threejs/resources/threejs/r132/examples/jsm/controls/OrbitControls.js';
import {GLTFLoader} from 'https://threejsfundamentals.org/threejs/resources/threejs/r132/examples/jsm/loaders/GLTFLoader.js';
import Stats from 'https://threejsfundamentals.org/threejs/resources/threejs/r132/examples/jsm/libs/stats.module.js'

var container, stats, controls, mixer, clock;
var camera, scene, renderer, canvas;

init();
animate();
onWindowResize();

function init() {
	// Create Scene
	container = document.createElement( 'div' );
	document.body.appendChild( container );
	// Set Camera
	camera = new THREE.PerspectiveCamera( 75, window.innerWidth / (window.innerHeight * 0.9), 0.5, 500 );
	camera.position.set( 0, 60, 120 );
	// Set Scene
	scene = new THREE.Scene();
	scene.background = new THREE.Color( 0x222222 );
	
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
	renderer.gammaOutput = true;
	container.appendChild( renderer.domElement );

	// Event Listener
	window.addEventListener( 'resize', onWindowResize, true );

	// Create controler
	controls = new OrbitControls( camera, renderer.domElement );
	controls.enableZoom = true
	controls.enableDamping = true
	controls.minDistance = 50;
	controls.maxDistance = 200;
	controls.target.set( 0, 0, 0 );
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
	/*const rtn = document.querySelector('#button0');
	rtn.addEventListener('click', () => {
		var port = location.port;
		window.open("http://127.0.0.1:" + port + "/index.html", "page")
	})
	const btn1 = document.querySelector('#button1');
	btn1.addEventListener('click', () => { // Send EEG
		camera.position.set(-30, 50, -60);
		window.open("http://" + location.hostname + ":" + location.port + "/eeg.html");
	})
	const btn3 = document.querySelector('#button2');
	btn3.addEventListener('click', () => { // Hospital
		camera.position.set(-60, 50, -30);
		window.open("http://" + location.hostname + ":" + location.port + "/map.html");
	})*/
}

function animate() {
    requestAnimationFrame(animate)
    renderer.render(scene, camera)
}

function onWindowResize() {
	camera.aspect = window.innerWidth / window.innerHeight;
	camera.updateProjectionMatrix();
	renderer.setSize( window.innerWidth, window.innerHeight );
}

function loadModel() {
    // Create Pie Chart
    const geometry1 = new THREE.CylinderGeometry( 50, 50, 10, 28, 1, false, 0, 1.2 * Math.PI );
    const material1 = new THREE.MeshBasicMaterial( {color: 0xffff00} );
    const cylinder1 = new THREE.Mesh( geometry1, material1 );
    scene.add( cylinder1 );
    const geometry2 = new THREE.CylinderGeometry( 50, 50, 10, 28, 1, false, 1.2 * Math.PI, 0.5 * Math.PI );
    const material2 = new THREE.MeshBasicMaterial( {color: 0x00ffff} );
    const cylinder2 = new THREE.Mesh( geometry2, material2 );
    scene.add( cylinder2 );
    const geometry3 = new THREE.CylinderGeometry( 50, 50, 10, 28, 1, false, 1.7 * Math.PI, 0.3 * Math.PI );
    const material3 = new THREE.MeshBasicMaterial( {color: 0xff00ff} );
    const cylinder3 = new THREE.Mesh( geometry3, material3 );
    scene.add( cylinder3 );

	// Create Text Geometry
	const loader = new THREE.FontLoader();
	loader.load( './static/assets/roboto/Roboto_Regular.json', function(font) {
		const matLite = new THREE.MeshBasicMaterial({
			color: 0x006699,
			transparent: true,
			opacity: 0.8,
			side: THREE.DoubleSide
		})

		const message = ["Depression\n60%", "Emotional Dysregulation\n15%", "Wonder\n25%"]
		const shapes1 = font.generateShapes(message[0], 5)
		const shapes2 = font.generateShapes(message[1], 5)
		const shapes3 = font.generateShapes(message[2], 5)
		const geometry1 = new THREE.ShapeGeometry(shapes1);
		const geometry2 = new THREE.ShapeGeometry(shapes2);
		const geometry3 = new THREE.ShapeGeometry(shapes3);
		geometry1.computeBoundingBox()
		geometry2.computeBoundingBox()
		geometry3.computeBoundingBox()

		const xMid1 = - 0.5 * ( geometry1.boundingBox.max.x - geometry1.boundingBox.min.x );
		const xMid2 = - 0.5 * ( geometry2.boundingBox.max.x - geometry2.boundingBox.min.x );
		const xMid3 = - 0.5 * ( geometry3.boundingBox.max.x - geometry3.boundingBox.min.x );
		geometry1.translate( xMid1, 0, 0 );
		geometry2.translate( xMid2, 0, 0 );
		geometry3.translate( xMid3, 0, 0 );

		const text1 = new THREE.Mesh( geometry1, matLite)
		const text2 = new THREE.Mesh( geometry2, matLite)
		const text3 = new THREE.Mesh( geometry3, matLite)

		text1.position.set(40, 15, 0)
		text1.rotateX(50)
		text2.position.set(-20, 15, 40)
		text3.position.set(-40, 15, -10)
		scene.add(text1)
		scene.add(text2)
		scene.add(text3)

	})
}