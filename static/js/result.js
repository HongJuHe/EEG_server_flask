import * as THREE from 'https://threejsfundamentals.org/threejs/resources/threejs/r132/build/three.module.js';
import {OrbitControls} from 'https://threejsfundamentals.org/threejs/resources/threejs/r132/examples/jsm/controls/OrbitControls.js';
import {GLTFLoader} from 'https://threejsfundamentals.org/threejs/resources/threejs/r132/examples/jsm/loaders/GLTFLoader.js';
import Stats from 'https://threejsfundamentals.org/threejs/resources/threejs/r132/examples/jsm/libs/stats.module.js'

var container, stats, controls, clock;
var camera, scene, renderer, geometry, text;
var ratioWidth = 0.7, ratioHeight = 0.9;

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
	camera = new THREE.PerspectiveCamera( 75, (window.innerWidth * ratioWidth) / (window.innerHeight * ratioHeight), 0.5, 1000 );
	camera.position.set( 0, 300, 300 );
	// Set Scene
	scene = new THREE.Scene();
	scene.background = new THREE.Color( 0xeeccaa );
	
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
	loadText('선택하세요', new THREE.Vector3(-50, 50, 0));
	
	// Set Renderer
	renderer = new THREE.WebGLRenderer( { antialias: true, alpha: true } );
	renderer.setSize( window.innerWidth * ratioWidth, window.innerHeight * ratioHeight );
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
	camera.aspect = (window.innerWidth * ratioWidth) / (window.innerHeight * ratioHeight );
	camera.updateProjectionMatrix();
	renderer.setSize( window.innerWidth * ratioWidth, window.innerHeight * ratioHeight );
}

function loadModel() {
	// GLTF Object with animation - mesh
	var loader = new GLTFLoader().setPath('../static/assets/Seoul/')
	loader.load('Seoul.glb', function(gltf) {
        var obj = gltf.scene;
		obj.position.set(-50, 0, -50)
		obj.children[0].children[0].material.color.set(0x095b09) // Darker
		scene.add( obj );
        obj.children.forEach( (child, ndx) => {
            objects.push(child)
        })
	})
	loader.load('Seoul.glb', function(gltf) {
        var obj = gltf.scene;
		obj.position.set(-50, 0, -50)
		obj.children[0].children[0].material.color.set(0x349943) // Lighter
		scene.add( obj );
	})
}

function loadText(string_name, string_loc) {
	if(text) {
		text.parent.remove(text)
	}

	// Create Text Geometry
	const loader = new THREE.FontLoader();
	loader.load( '../assets/Do_Hyeon/Do_Hyeon_Regular.json', function(font) {
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
	mouse.x = ( event.clientX / (window.innerWidth * ratioWidth) ) * 2 - 1;
	mouse.y = - ( event.clientY / (window.innerHeight * ratioHeight) - (window.innerHeight * 0.0001) ) * 2 + 1;

	raycaster.setFromCamera( mouse, camera );
	var intersections = raycaster.intersectObjects(objects, true);
	if ( intersections.length > 0 ) {
        if(object_selected) {
			if(object_selected == intersections[ 0 ].object.parent)
				return;
            object_selected.scale.setScalar(1)
            object_selected.position.add(new THREE.Vector3(0, -1, 0))
        }

		var object = intersections[ 0 ].object.parent;
        object_selected = object;
        object.scale.z *= 2;
        object.position.add(new THREE.Vector3(0, 1, 0))
		
		var name = object.name;
		for(let i = 0; i < objects.length; i++) {
			if(name == objects[i].name) {
				name = locations[i]
			}
		}
		var loc = new THREE.Vector3(object.position.x - 65, 10, object.position.z - 50);
		loadText(name, loc);

		getStatement(object.name);
	}
}

function getStatement(name) {
    const data = JSON.parse(JSON.stringify(Params));

    var description = "<hr>";
    const listInfo = document.getElementById("list-info");

    for (const key in Object.keys(data)) {
        const Gu = Object.keys(data)[key];
        
        if(Gu == name) {
            const names = Object.values(data[Gu])[0];
            const scores = Object.values(data[Gu])[1];
            const links = Object.values(data[Gu])[2];
            const addresses = Object.values(data[Gu])[3];
            
            var maxLength = names.length;

            var order = Object.keys(scores);
            for(var i = 0; i < maxLength - 1; i++) {
                for(var j = i+1; j < maxLength; j++) {
                    if(scores[i] < scores[j]) {
                        var tmp;    
                        tmp = scores[i];
                        scores[i] = scores[j];
                        scores[j] = tmp;

                        tmp = order[i];
                        order[i] = order[j];
                        order[j] = tmp;
                    }
                }
            }

            for(var i = 0; i < maxLength; i++) {
                var j = order[i];
                description += makeInformation(names[j], scores[i], links[j], addresses[j]);
            }

            listInfo.innerHTML = description;
        }       
    }
}

function makeInformation(name, score, link, address) {
    var string = ''

    string += '<h4>';
    string += name + '</h4>';

    string += '<h6>';
    string += address + '</h6>';

    var starCount = 5;
    while(score >= 1) {
        string += '<i class="fas fa-star fa-2x"></i>';
        score -= 1;
        starCount--;
    }
    if(score >= 0.5) {
        string += '<i class="fas fa-star-half-alt fa-2x"></i>';
        starCount--;
    }
    while(starCount > 0) {
        string += '<i class="far fa-star fa-2x"></i>';
        starCount--;
    }

    string += '<br>';
    if(link == "없습니다") {
        string += '<a class="btn btn-xl btn-light me-4" href="#!">';
        string += '병원 홈페이지 준비중</a>';
    } else {        
        string += '<a class="btn btn-dark btn-xl" href="';
        string += link + '" target="_blank">병원 홈페이지 바로가기</a>';
    }

    return string + '<p><br><br><br><br><hr color="navy">';
}