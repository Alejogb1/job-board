---
title: "Getting that Flat Shaded Look with Phong & Spotlights: Easy Trick!"
date: '2024-11-08'
id: 'getting-that-flat-shaded-look-with-phong-spotlights-easy-trick'
---

```javascript
var camera, scene, renderer;

init();
animate();

function init() {

  camera = new THREE.PerspectiveCamera(70, window.innerWidth / window.innerHeight, 0.01, 10);
  camera.position.z = 4;
  scene = new THREE.Scene();
  scene.add(camera);

  var ambientLight = new THREE.AmbientLight(0xcccccc, 0.2);
  scene.add(ambientLight);

  var spotLight = new THREE.SpotLight({ color: 0xffffff, angle: Math.PI / 10, intensity: 2 });
  spotLight.position.z = 1.5;
  scene.add(spotLight);

  var geometry = new THREE.SphereBufferGeometry( 1, 12, 16);
  var material = new THREE.MeshPhongMaterial({color: 0xff0000, flatShading: true, shininess: 100, specular: 0x00ff00});

  var mesh = new THREE.Mesh(geometry, material);
  scene.add(mesh);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  document.body.appendChild(renderer.domElement);

}

function animate() {
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
}
```
