import * as THREE from 'https://threejsfundamentals.org/threejs/resources/threejs/r132/build/three.module.js';
import {OrbitControls} from 'https://threejsfundamentals.org/threejs/resources/threejs/r132/examples/jsm/controls/OrbitControls.js';
import {GLTFLoader} from 'https://threejsfundamentals.org/threejs/resources/threejs/r132/examples/jsm/loaders/GLTFLoader.js';
import Stats from 'https://threejsfundamentals.org/threejs/resources/threejs/r132/examples/jsm/libs/stats.module.js'

var container, stats, controls, clock;
var camera, scene, renderer, geometry, text;
var ratioWidth = 1.0, ratioHeight = 0.9;

init();
animate();
onWindowResize();

function init() {
	// Create Scene
	container = document.createElement( 'div' );
	document.body.appendChild( container );
	// Set Camera
	camera = new THREE.PerspectiveCamera( 75, (window.innerWidth * ratioWidth) / (window.innerHeight * ratioHeight), 0.5, 2000 );
	camera.position.set( 0, 200, 300 );
	// Set Scene
	scene = new THREE.Scene();
	scene.background = new THREE.Color( 0xee99aa );
	
	clock = new THREE.Clock();

	// Light From Camera
	const hemiLight = new THREE.HemisphereLight( 0xffffff, 0x444444 );
	hemiLight.position.set( 0, 200, 0 );
	scene.add( hemiLight );
	// Light of Sun
	const dirLight = new THREE.DirectionalLight( 0xffffff );
	dirLight.position.set( 0, 0, 100 );
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
	renderer.setSize( window.innerWidth * ratioWidth, window.innerHeight * ratioHeight );
	renderer.setPixelRatio( Math.min(window.devicePixelRatio, 2) );
	renderer.gammaOutput = true;
	container.appendChild( renderer.domElement );

	// Event Listener
	window.addEventListener( 'resize', onWindowResize, true );

	// Create controler
	controls = new OrbitControls( camera, renderer.domElement );
	controls.enableZoom = true
	controls.enableDamping = true
	controls.minDistance = 200;
	controls.maxDistance = 300;
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
    requestAnimationFrame(animate)
    renderer.render(scene, camera)
}

function onWindowResize() {
	camera.aspect = (window.innerWidth * ratioWidth) / (window.innerHeight * ratioHeight );
	camera.updateProjectionMatrix();
	renderer.setSize( window.innerWidth * ratioWidth, window.innerHeight * ratioHeight );
}

function loadModel() {
	// GLTF Object with animation - Classroom
	var loader = new GLTFLoader().setPath('../static/assets/')
	loader.load('Class.glb', function(gltf) {
        var obj = gltf.scene;
		obj.position.set(0, 50, 300);
		obj.rotation.y = 1.5;
		scene.add( obj );
	})
	
	// import data from server
    // 3 Dementional Data from 'EEG_Wiset_Web'
    //var dp = {{ value|tojson }};
	var data = document.getElementById("eeg").innerHTML;
	data = data.replaceAll("'", '"')
	//console.log(data)
	//console.log(typeof(data))
	data = '{"eeg_data":' + data;
	data = data + '}'
	//console.log(data)
	var data_j = JSON.parse(data)
	//console.log(data_j.eeg_data[0])
    var data_eeg = data_j.eeg_data[0];
	var data_raw = data_j.eeg_data[1];
    var data_chat = data_j.eeg_data[2];

	// Load Pie Chart & Text - Depression
	var pieValues = [0, 0];
	for(let i = 0; i < data_eeg.length; i++) {
		if(data_eeg[i][1] == 1) {
			pieValues[1]++;
		} else {
			pieValues[0]++;
		}
	}
	var pieItems = ["우울증_아님", "우울증"]
	loadPieChart(pieValues,  pieItems);

	// Load Line Chart & Text
	loadLineChart(data_raw);

	// Load Bar Chart & Text
    var symptomNames_ = [ 'depression', 'sadness', 'lonely', 'angry', 'emotionaldysregulation',
	'lose', 'nervous', 'fatigue', 'confidence', 'lowesteem', 'despair', 'suicide']
    var symptomNames_kr = [ '우울함', '슬픔', '외로움', '화', '감정 조절 장애',
	'상실감', '긴장', '피로', '자신감 저하', '자존감 저하', '실망', '자살충동']

    var symptomDepression_ = ['depression', 'depression_gloomy', 'depression_dazed', 'depression_lethargy', 'depression_desire', 'depression_insomnia',
    'depression_interest', 'depression_appetite', 'depression_confidence', 'depression_loser', 'depression_concentration']
    var symptomSadness_ = ['sadness', 'sadness_upset', 'sadness_tear', 'sadness_cry', 'sadness_guilty', 'sadness_miss', 'sadness_remorse',
    'sadness_sad', 'sadness_miserable', 'sadness_vanity', 'sadness_despair', 'sadness_unfair', 'sadness_regret', 'sadness_disappointment']
    var symptomLonely_ = ['lonely', 'lonely_meaningless', 'lonely_suitability', 'lonely_smolder', 'lonely_hard' ]
    var symptomAngry_ = ['angry', 'angry_resentment', 'angry_dissatisfaction', 'angry_hate', 'angry_anger', 'angry_hatred']
    var symptomEmotionaldysregulation_ = [ 'emotionaldysregulation_paralysis', 'emotionaldysregulation_arbitrariness',
        'emotionaldysregulation_suppression', 'emotionaldysregulation_conflict']
	var symptomLose_ = ['lose', 'lose_vacuity', 'lose_novalue', 'lose_collapse', 'lose_lose', 'lose_loneliness']
	var symptomNervous_ = ['nervous', 'nervous_restlessness', 'nervous_nervous', 'nervous_notstay', 'nervous_sharp',
	'nervous_petulance', 'nervous_confuse', 'nervous_tremble', 'nervous_frustrated', 'nervous_breath',
	'nervous_updown', 'nervous_worry']
	var symptomFatigue_ = ['fatigue', 'fatigue_slump', 'fatigue_sleepy', 'fatigue_oversleep']
	var symptomConfidence_ = ['confidence', 'confidence_indecisive', 'confidence_atrophy', 'confidence_timid', 'confidence_decide']
	var symptomLowesteem_ = ['lowesteem', 'lowesteem_insignificant', 'lowesteem_ugly', 'lowesteem_derision',
		'lowesteem_look', 'lowesteem_frustration', 'lowesteem_watch', 'lowesteem_retribution',
		'lowesteem_finger', 'lowesteem_stare', 'lowesteem_damage', 'lowesteem_heartbreak']
	var symptomDespair_ = ['despair', 'despair_nohope', 'despair_giveup', 'despair_suffer', 'despair_disapoint',
		'despair_insult', 'despair_weak', 'despair_imminence']
	var symptomSuicide_ = ['suicide']

    var symptomDepression_kr = ['우울함', '침울함', '멍함', '무기력', '의욕_없음', '불면증',
    '흥미_없음', '식욕_없음', '자존감_하락', '패배감', '집중력_하락']
    var symptomSadness_kr = ['슬픔', '속상', '눈물', '울음', '죄책감', '그리움', '연민',
    '서러움', '비참함', '허망함', '절망', '억울함', '후회', '서운함']
    var symptomLonely_kr = ['외로움', '무의미', '적적함', '울적', '고단함' ]
    var symptomAngry_kr = ['화', '원망', '불만', '미움', '분노', '증오']
    var symptomEmotionaldysregulation_kr = ['감정_마비', '독단', '강압', '갈등']
	var symptomLose_kr = ['상실감', '공허', '무가치', '허탈함', '자괴감', '외로움']
	var symptomNervous_kr = ['긴장', '초조함', '조마조마함', '압박감', '예민함',
	'짜증', '혼란', '떨림', '답답함', '과호흡',	'감정기복', '걱정']
	var symptomFatigue_kr = ['피로', '슬럼프', '졸림', '과수면']
	var symptomConfidence_kr = ['자신감_저하', '우유부단함', '위축', '소심', '결단력_부족']
	var symptomLowesteem_kr = ['자존감_저하', '하찮음', '열등함', '업신여김',
		'눈초리', '좌절', '감시', '보복', '손가락질', '노려봄', '피해의식', '한심']
	var symptomDespair_kr = ['절망감', '희망_없음', '포기', '고통', '실망',
		'모욕', '나약', '절박']
	var symptomSuicide_kr = ['자살충동']
			
	var symptomDepression_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    var symptomSadness_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    var symptomLonely_count = [0, 0, 0, 0, 0 ]
    var symptomAngry_count = [0, 0, 0, 0, 0, 0]
    var symptomEmotionaldysregulation_count = [ 0, 0, 0, 0]
    var symptomLose_count = [ 0, 0, 0, 0, 0, 0]
	var symptomNervous_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	var symptomFatigue_count = [0, 0, 0, 0]
	var symptomConfidence_count = [0, 0, 0, 0, 0]
	var symptomLowesteem_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	var symptomDespair_count = [0, 0, 0, 0, 0, 0, 0, 0]
	var symptomSuicide_count = [0]

    for(let i = 0; i < data_chat.length; i++) {
        for(let j = 0; j < data_chat[i].length; j++) {
            const symptom = data_chat[i][j];
            for(let k = 0; k < symptomNames_.length; k++) {
                if(symptom.indexOf(symptomNames_[k]) == 0) {
                    let num = 0;
                    switch(k) {
                        case 0: // 우울증
                            num = symptomDepression_.indexOf(symptom);
                            symptomDepression_count[num]++;
                            break;
                        case 1: // 슬픔
                            num = symptomSadness_.indexOf(symptom);
                            symptomSadness_count[num]++;
                            break;
                        case 2: // 외로움
                            num = symptomLonely_.indexOf(symptom);
                            symptomLonely_count[num]++;
                            break;
                        case 3: // 분노
                            num = symptomAngry_.indexOf(symptom);
                            symptomAngry_count[num]++;
                            break;
						case 4: // 감정 조절 장애
							num = symptomEmotionaldysregulation_.indexOf(symptom);
							symptomEmotionaldysregulation_count[num]++;
							break;
						case 5: // 상실감
							num = symptomLose_.indexOf(symptom);
							symptomLose_count[num]++;
							break;
						case 6: // 긴장
							num = symptomNervous_.indexOf(symptom);
							symptomNervous_count[num]++;
							break;
						case 7: // 피로
							num = symptomFatigue_.indexOf(symptom);
							symptomFatigue_count[num]++;
							break;
						case 8: // 자신감 저하
							num = symptomConfidence_.indexOf(symptom);
							symptomConfidence_count[num]++;
							break;
						case 9: // 자존감 저하
							num = symptomLowesteem_.indexOf(symptom);
							symptomLowesteem_count[num]++;
							break;
						case 10: // 실망
							num = symptomDespair_.indexOf(symptom);
							symptomDespair_count[num]++;
							break;
						case 11: // 자살충동
							num = symptomSuicide_.indexOf(symptom);
							symptomSuicide_count[num]++;
							break;
                    }

                    break;
                }
            }
        }
    }
	
	var xPos1 = 500, xPos2 = -500, yPos = -250;
	loadBarChart1(symptomDepression_count, symptomDepression_kr, new THREE.Vector3(xPos1, yPos, -250));
	loadBarChart1(symptomSadness_count, symptomSadness_kr, new THREE.Vector3(xPos1, yPos, -100));
	loadBarChart1(symptomLonely_count.concat(symptomAngry_count), symptomLonely_kr.concat(symptomAngry_kr), new THREE.Vector3(xPos1, yPos, 50));
	loadBarChart1(symptomEmotionaldysregulation_count.concat(symptomLose_count), symptomEmotionaldysregulation_kr.concat(symptomLose_kr), new THREE.Vector3(xPos1, yPos, 200));

	loadBarChart2(symptomNervous_count, symptomNervous_kr, new THREE.Vector3(xPos2, yPos, -250));
	loadBarChart2(symptomFatigue_count.concat(symptomConfidence_count), symptomFatigue_kr.concat(symptomConfidence_kr), new THREE.Vector3(xPos2, yPos, -100));
	loadBarChart2(symptomLowesteem_count, symptomLowesteem_kr, new THREE.Vector3(xPos2, yPos, 50));
	loadBarChart2(symptomDespair_count.concat(symptomSuicide_count), symptomDespair_kr.concat(symptomSuicide_kr), new THREE.Vector3(xPos2, yPos, 200));
}

function loadPieChart(values, items) {
	const pie_radius = 100;
	const pie_height = 10;
	const pie_segment = 28;
	
	var sum = 0;
	for(let i = 0; i < values.length; i++) {
		sum += values[i];
	}

	const colors = [0x123456, 0x789abc]
	var start_value = 0;
	for(let i = 0; i < values.length; i++) {
		const value = (values[i] / sum * 2) * Math.PI;
		const geometry = new THREE.CylinderGeometry( pie_radius, pie_radius, pie_height, pie_segment, 1, false, start_value, value );
		const material = new THREE.MeshBasicMaterial( {color: colors[i]} );
		const cylinder = new THREE.Mesh( geometry, material );
		cylinder.position.set(0, 0, 400);
		cylinder.rotation.x += Math.PI / 2;
		cylinder.rotation.z -= Math.PI;
		scene.add( cylinder );
		const pos = new THREE.Vector3(cylinder.position.x - pie_radius, cylinder.position.y + 10 + i * 25, cylinder.position.z);
		loadText(items[i], pos, Math.PI, colors[i]);
		start_value += value;
	}
}

function loadBarChart1(values, items, location) {
	const barUnit = 10;
	const barMulti = 30;

	for(let i = 0; i < values.length; i++) {
		const geometry = new THREE.BoxGeometry( barUnit, values[i] * barMulti, barUnit );
		const color = Math.random() * 0xbbbbbb + 0x444444;
		const material = new THREE.MeshBasicMaterial( {color: color} );
		const cube = new THREE.Mesh( geometry, material );
		cube.position.set(location.x, values[i] * barMulti / 2 + location.y,  barUnit*i + location.z);
		scene.add( cube );

		const pos = new THREE.Vector3(location.x, i * 25 + location.y + 200, location.z);
		loadText(items[i], pos, -Math.PI / 2, color);
	}
	
}
function loadBarChart2(values, items, location) {
	const barUnit = 10;
	const barMulti = 30;

	for(let i = 0; i < values.length; i++) {
		const geometry = new THREE.BoxGeometry( barUnit, values[i] * barMulti, barUnit );
		const color = Math.random() * 0xdddddd + 0x000000;
		const material = new THREE.MeshBasicMaterial( {color: color} );
		const cube = new THREE.Mesh( geometry, material );
		cube.position.set(location.x, values[i] * barMulti / 2 + location.y, location.z - barUnit*i);
		scene.add( cube );

		const pos = new THREE.Vector3(location.x, i * 25 + location.y + 200, location.z);
		loadText(items[i], pos, Math.PI / 2, color);
	}
	
}

function loadLineChart(values) {
	const lineUnit = 0.6;
	const pointUnit = 15;
	const points = []; 
	for(let i = 0; i < values.length; i++) {
		points.push(new THREE.Vector3(-250 + lineUnit * i, values[i] * pointUnit, -390));
	}
	
	const geometry = new THREE.BufferGeometry().setFromPoints( points );
	const material = new THREE.MeshBasicMaterial( {color: 0xffffff} );
	const line = new THREE.Line( geometry, material );
	scene.add( line );
	
	const pos = new THREE.Vector3(0, 100, -400);
	loadText("뇌파의_파형", pos, 0, 0xffffff);
}


function loadText(string_name, string_loc, string_rot, string_color) {
	if(text) {
		text.parent.remove(text)
	}

	// Create Text Geometry
	const loader = new THREE.FontLoader();
	loader.load( '../static/assets/Do_Hyeon/Do_Hyeon_Regular.json', function(font) {
		geometry = new THREE.TextGeometry(string_name, {
			font: font,
			size: 20,
			height: 2,
		})
		text = new THREE.Mesh(geometry, [
			new THREE.MeshPhongMaterial({ color: string_color }), //font
			new THREE.MeshPhongMaterial({ color: string_color }) //side
		])
		text.castShadow = true
		text.position.set(string_loc.x, string_loc.y, string_loc.z)
		text.rotateY(string_rot)
		scene.add(text)
	})
}