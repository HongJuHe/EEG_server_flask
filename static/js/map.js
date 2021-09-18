import * as THREE from 'https://threejsfundamentals.org/threejs/resources/threejs/r132/build/three.module.js';
import {OrbitControls} from 'https://threejsfundamentals.org/threejs/resources/threejs/r132/examples/jsm/controls/OrbitControls.js';
import {GLTFLoader} from 'https://threejsfundamentals.org/threejs/resources/threejs/r132/examples/jsm/loaders/GLTFLoader.js';
import Stats from 'https://threejsfundamentals.org/threejs/resources/threejs/r132/examples/jsm/libs/stats.module.js'

//import {FBXLoader} from '../three.js-master/examples/jsm/loaders/FBXLoader.js'
//import {DRACOLoader} from '../three.js-master/examples/jsm/loaders/DRACOLoader.js'
//import vShader from '../shaders/vertexShader.glsl.js'
//import fShader from '../shaders/fragmentShader.glsl.js'
//import { TetrahedronBufferGeometry } from '../three.js-master/build/three.module.js';

var container, stats, controls, clock;
var camera, scene, renderer, geometry, text;

var objects = [], object_selected;
var mouse = new THREE.Vector2(), raycaster = new THREE.Raycaster();
var locations = ['강동구', '중랑구', '노원구', '도봉구', '강북구', '성북구',
'동대문구', '광진구', '성동구', '종로구', '은평구', '서대문구',
'중구', '용산구', '송파구', '강남구', '서초구', '마포구',
'강서구', '영등포구', '양천구', '구로구', '동작구', '관악구', '금천구']

init();
animate();
onWindowResize();

function init() {
	// Create Scene
	container = document.createElement( 'div' );
	document.body.appendChild( container );
	// Set Camera
	camera = new THREE.PerspectiveCamera( 75, (window.innerWidth * 0.5) / (window.innerHeight * 0.9), 0.5, 1000 );
	camera.position.set( 0, 300, 300 );
	// Set Scene
	scene = new THREE.Scene();
	//scene.background = new THREE.Color( 0x349934 );
	
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
	loadText('지역을\n선택하세요.', new THREE.Vector3(0, 50, 0));
	
	// Set Renderer
	renderer = new THREE.WebGLRenderer( { antialias: true, alpha: true } );
	renderer.setSize( window.innerWidth * 0.5, window.innerHeight * 0.9 );
	renderer.setPixelRatio( Math.min(window.devicePixelRatio, 2) );
	renderer.setAnimationLoop()
	renderer.gammaOutput = true;
	container.appendChild( renderer.domElement );

	// Event Listener
	window.addEventListener( 'resize', onWindowResize, true );
	document.addEventListener( 'mousedown', onDocumentMouseDown, false );

	// Create controler
	controls = new OrbitControls( camera, renderer.domElement );
	controls.enableZoom = true
	controls.enableDamping = true
	controls.minDistance = 200;
	controls.maxDistance = 500;
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
}

function animate() {
	requestAnimationFrame( animate );
	renderer.render( scene, camera );
	stats.update();
}

function onWindowResize() {
	camera.aspect = (window.innerWidth * 0.5) / (window.innerHeight * 0.9);
	camera.updateProjectionMatrix();
	renderer.setSize( window.innerWidth * 0.5, window.innerHeight * 0.9 );
}

function loadModel() {
	// GLTF Object with animation - mesh
	var loader = new GLTFLoader().setPath('./static/assets/Seoul/')
	loader.load('Seoul.glb', function(gltf) {
        var obj = gltf.scene;
		obj.position.set(-50, 0, -50)
		obj.children[0].children[0].material.color.set(0x125612) // Darker
		scene.add( obj );
        obj.children.forEach( (child, ndx) => {
            objects.push(child)
        })
	})
	loader.load('Seoul.glb', function(gltf) {
        var obj = gltf.scene;
		obj.position.set(-50, 0, -50)
		obj.children[0].children[0].material.color.set(0x349934) // Lighter
		scene.add( obj );
	})
}

function loadText(string_name, string_loc) {
	if(text) {
		text.parent.remove(text)
	}

	// Create Text Geometry
	const loader = new THREE.FontLoader();
	loader.load( './static/assets/Do_Hyeon/Do_Hyeon_Regular.json', function(font) {
		geometry = new THREE.TextGeometry(string_name, {
			font: font,
			size: 15,
			height: 2,
		})
		text = new THREE.Mesh(geometry, [
			new THREE.MeshPhongMaterial({ color: 0xad4000 }), //font
			new THREE.MeshPhongMaterial({ color: 0x5c2301 }) //side
		])
		text.castShadow = true
		text.position.set(string_loc.x, string_loc.y, string_loc.z)
		text.rotateX(50)
		scene.add(text)
	})
}

function onDocumentMouseDown(event) {
	event.preventDefault();
	mouse.x = ( event.clientX / (window.innerWidth * 0.5) ) * 2 - 1;
	mouse.y = - ( event.clientY / (window.innerHeight * 0.9) - (window.innerHeight * 0.0001) ) * 2 + 1;

	raycaster.setFromCamera( mouse, camera );
	var intersections = raycaster.intersectObjects(objects, true);
	if ( intersections.length > 0 ) {
        if(object_selected) {
            object_selected.scale.setScalar(1)
            object_selected.position.add(new THREE.Vector3(0, -1, 0))
        }

		var object = intersections[ 0 ].object.parent;
        object_selected = object;
        object.scale.z *= 2;
        object.position.add(new THREE.Vector3(0, 1, 0))
		console.log(object) // print LOCATION INFO
		
		var name = object.name;
		for(let i = 0; i < objects.length; i++) {
			if(name == objects[i].name) {
				name = locations[i]
			}
		}
		var loc = new THREE.Vector3(object.position.x - 65, 10, object.position.z - 50);
		loadText(name, loc);
	}
}