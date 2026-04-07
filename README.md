# 🛸 DroneTrack-CV

### *Once upon a time, a drone looked down at the world and wanted to understand what it saw...*


![m210_stereo_pt_cloud-eeed34d8dc](https://github.com/user-attachments/assets/ad96c97d-c359-409c-af64-cb35150238ac)



---

## 🎬 The Story

Imagine you're a drone, flying high above a busy city. You can see tiny cars moving, people walking, cyclists pedaling. But you're just watching — you can't *understand* what you see.

**DroneTrack-CV gives drones eyes that think.**

It watches a live video feed from above, spots every vehicle, person, and cyclist in the frame, gives each one a name tag (like a little ID badge), and follows them around — even if they hide behind a tree for a moment and come back.

---

## 🧩 How It Works (The Simple Version)

```
📷 Drone camera sees the world
        ↓
🔍 YOLOv8 says "that's a car, that's a person!"
        ↓
🏷️ DeepSORT says "that car is #7, I saw it last frame too"
        ↓
🎨 We draw pretty boxes and trails on the video
        ↓
📹 You get a cool annotated video + data export
```

That's it. Four steps. Camera → Detect → Track → Draw.

---

## 🚀 Try It Yourself

**Step 1:** Install the ingredients
```bash
pip install -r requirements.txt
```

**Step 2:** Feed it a video
```bash
python main.py --source your_drone_video.mp4 --show-trails --show-minimap --show-hud
```

**Step 3:** Watch the magic happen ✨

A window pops up showing your drone video with colored boxes around every object, little trails showing where they've been, and a minimap in the corner like a video game.

---

## 🎨 What You'll See

- 🟦 **Cyan boxes** → Vehicles (cars, trucks, buses)
- 🟧 **Orange boxes** → People walking around
- 🟨 **Yellow boxes** → Cyclists
- 🟥 **Red boxes** → Something unusual (anomaly!)
- 🟣 **Colorful trails** → Where each object has been moving
- 🗺️ **Minimap** → Bird's-eye view of all tracked objects

---

## 📂 What's Inside

```
DroneTrack-CV/
├── main.py              ← Start here. This runs everything.
├── src/
│   ├── detector.py      ← The "eyes" — finds objects in each frame
│   ├── tracker.py       ← The "memory" — remembers who is who
│   ├── visualizer.py    ← The "artist" — draws boxes and trails
│   ├── hud.py           ← The "dashboard" — shows FPS and stats
│   ├── zone_alert.py    ← The "guard" — alerts if someone enters a zone
│   ├── exporter.py      ← The "reporter" — saves everything to CSV/JSON
│   └── pipeline.py      ← The "conductor" — connects all the pieces
├── configs/
│   └── default.yaml     ← Settings you can tweak
├── demo/
│   └── dronetrack-cv.html  ← Pretty interactive demo (open in browser!)
└── tests/               ← Making sure nothing is broken
```

---

## 🎯 Built For

This project is designed for [Sky-Drones](https://sky-drones.com/) — a company that builds smart drone systems for surveying, security, and inspection. DroneTrack-CV is the computer vision brain that makes their drones *see and understand*.

---

## 🌟 The End

And so, the little drone flew happily ever after — no longer just watching the world, but truly *seeing* it.

---

<p align="center">
  Built  by <a href="https://github.com/SamSon1402">Sameer</a>
</p>
