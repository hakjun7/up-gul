<template>
  <div class="flex items-center justify-center h-screen bg-gray-200">
    <div class="flex flex-col w-full h-full max-w-lg shadow bg-yellow-100 p-4">
      <div class="flex flex-row items-center">
        <div class="flex flex-row items-center justify-between mb-3">
          <img class="h-12" src="../assets/ubgul.png" alt="" />
        </div>
        <div class="relative inline-block ml-auto">
          <button class="focus:outline-none" v-if="!isOwner" @click="friendReq">
            <svg
              class="focus:outline-none h-8 mr-1 text-blue-800"
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 20 20"
              fill="currentColor"
            >
              <path
                d="M8 9a3 3 0 100-6 3 3 0 000 6zM8 11a6 6 0 016 6H2a6 6 0 016-6zM16 7a1 1 0 10-2 0v1h-1a1 1 0 100 2h1v1a1 1 0 102 0v-1h1a1 1 0 100-2h-1V7z"
              />
            </svg>
          </button>
          <button class="focus:outline-none" @click="backToHome">
            <svg
              class="h-8 text-blue-800"
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                stroke-linecap="round"
                stroke-linejoin="round"
                stroke-width="2"
                d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6"
              />
            </svg>
          </button>
        </div>
      </div>
      <div class="mb-8 mt-8">
        <div class="rel">
          <div class="blank"></div>
          <video
            width="320"
            height="240"
            id="localVideo"
            muted
            autoplay
            class="video_box"
          ></video>
          <video
            width="320"
            height="240"
            id="remoteVideo"
            autoplay
            class="video_box"
          ></video>
          <img
            class="video_box bg-yellow-100"
            src="@/assets/background.png"
            alt="background"
          />
          <img
            v-if="expression === 'neutral' && otherGender == 'male'"
            class="video_box bg-yellow-100"
            src="@/assets/mneutral.png"
            alt="neutral"
          />
          <img
            v-if="expression === 'negative' && otherGender == 'male'"
            class="video_box bg-yellow-100"
            src="@/assets/mnegative.png"
            alt="negative"
          />
          <img
            v-if="expression === 'positive' && otherGender == 'male'"
            class="video_box bg-yellow-100"
            src="@/assets/mpositive.png"
            alt="positive"
          />
          <img
            v-if="expression === 'neutral' && otherGender == 'female'"
            class="video_box bg-yellow-100"
            src="@/assets/wneutral.png"
            alt="neutral"
          />
          <img
            v-if="expression === 'negative' && otherGender == 'female'"
            class="video_box bg-yellow-100"
            src="@/assets/wnegative.png"
            alt="negative"
          />
          <img
            v-if="expression === 'positive' && otherGender == 'female'"
            class="video_box bg-yellow-100"
            src="@/assets/wpositive.png"
            alt="positive"
          />
          <img
            v-if="expression === 'question'"
            class="video_box bg-yellow-100"
            src="@/assets/waiting.png"
            alt="waiting"
          />
        </div>
      </div>
      <div v-if="noimo" class="h-48">
        <lottie v-if="noimo" :options="animationOptions" />
      </div>
      <div class="mt-5">
        <ul
          v-chat-scroll
          class="m-0 px-2 py-2 overflow-x-auto bg-blue-100"
          style="height: 30vh"
        >
          <li v-for="(message, idx) in messages" :key="idx">
            <div
              class="flex-col m-0 bg-blue-100"
              :class="{ mychat: message.nickname === nickname }"
            >
              <div
                class="border-gray-800 text-gray-800 rounded-full flex-start p-0"
                :class="{
                  chat_left: message.nickname === nickname,
                  chat_right: message.nickname !== nickname,
                }"
              >
                {{ message.nickname }}
              </div>
              <div
                class="flex justify-start mb-1"
                v-if="message.nickname === nickname"
              >
                <p class="bg-gray-400 py-1 px-2 rounded-lg">
                  {{ message.message }}
                </p>
                <span class="mx-1 self-end text-xs">
                  {{ message.timestamp }}
                </span>
              </div>

              <div class="flex justify-end mb-1" v-else>
                <span class="mx-1 mt-2 flex self-end text-xs">
                  {{ message.timestamp }}
                </span>
                <p class="bg-gray-400 py-1 px-2 rounded-lg">
                  {{ message.message }}
                </p>
              </div>
              <!-- <span :class="{ chat_right: message.nickname === nickname }">{{
                message.timestamp
              }}</span> -->
            </div>
          </li>
        </ul>
      </div>
      <div class="flex flex-col relative mt-1">
        <input
          class="rounded-r-md pl-3 h-10 w-full border border-white focus:outline-none bg-gray-100 placeholder-gray-500 focus:border-blue-800 focus:shadow-lg focus:bg-white"
          v-model="chatmessage"
          placeholder="메세지를 입력하세요"
          @keypress.enter="sendMessage(false)"
        />
        <button
          @click="sendMessage(false)"
          class="focus:outline-none bg-blue-800 absolute flex rounded-r-md items-center justify-center h-10 w-12 right-0 top-0"
        >
          <!-- ⤶ -->
          <svg
            class="h-6 w-6 text-yellow-500"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M3 10h10a8 8 0 018 8v2M3 10l6 6m-6-6l6-6"
            />
          </svg>
        </button>
      </div>
      <!-- <button
        @click="openMedia"
        class="bg-blue-500 hover:bg-blue-700 mt-5 text-white font-bold py-2 px-4 border border-blue-700 rounded"
      >
        카메라 / 오디오 열기
      </button>
      <button
        @click="initRoom"
        class="bg-blue-500 hover:bg-blue-700 mt-5 text-white font-bold py-2 px-4 border border-blue-700 rounded"
      >
        통신 시작
      </button>
      <button
        @click="captureAPI"
        class="bg-blue-500 hover:bg-blue-700 mt-5 text-white font-bold py-2 px-4 border border-blue-700 rounded"
      >
        감정 분석
      </button> -->
    </div>
  </div>
</template>

<script>
import { mapState } from "vuex";
import { parseRoomLink } from "@/utils.js";
// import { timestamptotime } from "@/utils.js";
import Lottie from "@/components/lottie.vue";
import * as sadEmoji from "@/assets/Emoji/sad.json";
import firebase from "firebase";
import axios from "axios";
import Vue from "vue";
import VueMoment from "vue-moment";
import VueMomentJs from "vue-momentjs";
Vue.use(VueMoment, VueMomentJs);

export default {
  computed: mapState(["uid", "signalConf", "nickname"]),

  async mounted() {
    // this.check();
    this.initChat();
    this.openMedia(true);
    // this.initRoom();
    this.sendHi();
  },
  components: {
    lottie: Lottie,
  },
  data() {
    return {
      repeatAPI: null,
      animationOptions: { animationData: sadEmoji.default },
      noimo: false,
      // webRTC data
      isOwner: null,
      peerConnection: null,
      localStream: null,
      remoteStream: null,
      configuration: {
        iceServers: [
          {
            urls: [
              "stun:stun1.l.google.com:19302",
              "stun:stun2.l.google.com:19302",
            ],
          },
          // turn 서버 추가
          {
            urls: "turn:numb.viagenie.ca",
            credential: "rlaxodls",
            username: "uutaein@gmail.com",
          },
        ],
      },
      roomId: this.$route.params.id,
      other: null,
      otherGender: null,
      // 채팅
      messages: null,
      chatmessage: "",
      scroll: true,
      expression: "question",
      time: "",
    };
  },
  methods: {
    backToHome() {
      this.captureAPI(false);
      this.deleteRoom();
      setTimeout(() => {
        location.reload();
      }, 10);
      this.$router.push({ name: "Main" });
    },
    async deleteRoom() {
      if (this.isOwner) {
        // firestore 삭제 근데 이거 할 필요 있나 싶긴함
        let db = firebase.firestore();
        let roomRef = db.collection("rooms").doc(this.roomId);
        roomRef.delete();

        // realtime database
        let targetRoom = firebase
          .database()
          .ref()
          .child("Rooms")
          .child(this.roomId);
        targetRoom.remove();
      }
    },
    async openMedia(once) {
      // const stream = await navigator.mediaDevices.getUserMedia({
      //   video: true,
      //   audio: true,
      // });
      // document.querySelector("#localVideo").srcObject = stream;
      // this.localStream = stream;
      // this.remoteStream = new MediaStream();
      // document.querySelector("#remoteVideo").srcObject = this.remoteStream;
      // console.log("stream", document.querySelector("#localVideo").srcObject);
      // this.captureAPI(true);
      if (once) {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: true,
        });
        document.querySelector("#localVideo").srcObject = stream;
        this.localStream = stream;
        this.remoteStream = new MediaStream();
        document.querySelector("#remoteVideo").srcObject = this.remoteStream;
        console.log("stream", document.querySelector("#localVideo").srcObject);
        this.initRoom();
      } else {
        this.captureAPI(true);
      }
    },
    initRoom() {
      if (parseRoomLink(this.$route.params.id) === this.uid) {
        console.log("방 주인");
        this.isOwner = true;
        this.createRTC();
      } else {
        console.log("방주인 아님");
        this.isOwner = false;
        this.joinRTC();
      }
    },
    async createRTC() {
      const db = this.$firebase.firestore();
      const roomRef = db.collection("rooms").doc(this.$route.params.id);

      console.log(
        "Create PeerConnection with configuration: ",
        this.configuration
      );
      this.peerConnection = new RTCPeerConnection(this.configuration);

      this.registerPeerConnectionListeners();

      // peerconnection 에 localstream 을 싣음.
      this.localStream.getTracks().forEach((track) => {
        this.peerConnection.addTrack(track, this.localStream);
      });

      // 연결이 올때까지 대기하는 것 같음.
      // Code for collecting ICE candidates below
      const callerCandidatesCollection = roomRef.collection("callerCandidates");

      this.peerConnection.addEventListener("icecandidate", (event) => {
        if (!event.candidate) {
          console.log("Got final candidate!");
          return;
        }
        console.log("Got candidate: ", event.candidate);
        callerCandidatesCollection.add(event.candidate.toJSON());
      });
      // Code for collecting ICE candidates above

      // Code for creating a room below
      const offer = await this.peerConnection.createOffer();
      await this.peerConnection.setLocalDescription(offer);
      console.log("Created offer:", offer);

      const roomWithOffer = {
        offer: {
          type: offer.type,
          sdp: offer.sdp,
        },
      };
      await roomRef.set(roomWithOffer);
      // this.roomId = roomRef.id;
      console.log(`New room created with SDP offer. Room ID: ${roomRef.id}`);
      // Code for creating a room above

      this.peerConnection.addEventListener("track", (event) => {
        console.log("Got remote track:", event.streams[0]);
        document.querySelector("#remoteVideo").srcObject = event.streams[0];
        this.remoteStream = event.streams[0];
        // event.streams[0].getTracks().forEach(track => {
        //   console.log('Add a track to the remoteStream:', track);
        //   this.remoteStream.addTrack(track);
        // });
      });

      // Listening for remote session description below
      roomRef.onSnapshot(async (snapshot) => {
        const data = snapshot.data();
        if (
          !this.peerConnection.currentRemoteDescription &&
          data &&
          data.answer
        ) {
          console.log("Got remote description: ", data.answer);
          const rtcSessionDescription = new RTCSessionDescription(data.answer);
          await this.peerConnection.setRemoteDescription(rtcSessionDescription);
        }
      });
      // Listening for remote session description above

      // Listen for remote ICE candidates below
      roomRef.collection("calleeCandidates").onSnapshot((snapshot) => {
        snapshot.docChanges().forEach(async (change) => {
          if (change.type === "added") {
            let data = change.doc.data();
            console.log(
              `Got new remote ICE candidate: ${JSON.stringify(data)}`
            );
            await this.peerConnection.addIceCandidate(
              new RTCIceCandidate(data)
            );
          }
        });
      });
      // Listen for remote ICE candidates above
      // this.captureAPI(true);
      this.openMedia(false);
    },
    async joinRTC() {
      const db = firebase.firestore();
      const roomRef = db.collection("rooms").doc(this.$route.params.id);
      const roomSnapshot = await roomRef.get();
      console.log("Got room:", roomSnapshot.exists);

      if (roomSnapshot.exists) {
        console.log(
          "Create PeerConnection with configuration: ",
          this.configuration
        );
        this.peerConnection = new RTCPeerConnection(this.configuration);
        this.registerPeerConnectionListeners();
        this.localStream.getTracks().forEach((track) => {
          this.peerConnection.addTrack(track, this.localStream);
        });

        // Code for collecting ICE candidates below
        const calleeCandidatesCollection = roomRef.collection(
          "calleeCandidates"
        );
        this.peerConnection.addEventListener("icecandidate", (event) => {
          if (!event.candidate) {
            console.log("Got final candidate!");
            return;
          }
          console.log("Got candidate: ", event.candidate);
          calleeCandidatesCollection.add(event.candidate.toJSON());
        });
        // Code for collecting ICE candidates above

        this.peerConnection.addEventListener("track", (event) => {
          console.log("Got remote track:", event.streams[0]);
          document.querySelector("#remoteVideo").srcObject = event.streams[0];
          this.remoteStream = event.streams[0];
          // event.streams[0].getTracks().forEach(track => {
          //   console.log('Add a track to the remoteStream:', track);
          //   this.remoteStream.addTrack(track);
          // });
        });

        // Code for creating SDP answer below
        const offer = roomSnapshot.data().offer;
        console.log("Got offer:", offer);
        await this.peerConnection.setRemoteDescription(
          new RTCSessionDescription(offer)
        );
        const answer = await this.peerConnection.createAnswer();
        console.log("Created answer:", answer);
        await this.peerConnection.setLocalDescription(answer);

        const roomWithAnswer = {
          answer: {
            type: answer.type,
            sdp: answer.sdp,
          },
        };
        await roomRef.update(roomWithAnswer);
        // Code for creating SDP answer above

        // Listening for remote ICE candidates below
        roomRef.collection("callerCandidates").onSnapshot((snapshot) => {
          snapshot.docChanges().forEach(async (change) => {
            if (change.type === "added") {
              let data = change.doc.data();
              console.log(
                `Got new remote ICE candidate: ${JSON.stringify(data)}`
              );
              await this.peerConnection.addIceCandidate(
                new RTCIceCandidate(data)
              );
            }
          });
        });
        // Listening for remote ICE candidates above
      }
      // this.captureAPI(true);
      this.openMedia(false);
    },
    async initChat() {
      const chat = await this.$db.ref("Messages/" + this.$route.params.id);
      chat.on("value", (snapshot) => {
        this.messages = [];
        snapshot.forEach((doc) => {
          const item = doc.val();
          item.key = doc.key;
          this.messages.push(item);
        });
      });
    },
    async sendHi() {
      setTimeout(() => {
        this.sendMessage(true);
      }, 500);
    },
    async sendMessage(init) {
      if (init) {
        this.chatmessage = "안녕하세요";
      }
      const messageRoomRef = await this.$db.ref(
        "Messages/" + this.$route.params.id
      );
      const messageRef = messageRoomRef.push();
      messageRef.set({
        uid: this.uid,
        message: this.chatmessage,
        nickname: this.nickname,
        timestamp: this.$moment(new Date()).format("HH:mm"),
      });
      this.chatmessage = "";
    },
    registerPeerConnectionListeners() {
      this.peerConnection.addEventListener("icegatheringstatechange", () => {
        console.log(
          `ICE gathering state changed: ${this.peerConnection.iceGatheringState}`
        );
      });

      this.peerConnection.addEventListener("connectionstatechange", () => {
        console.log(
          `Connection state change: ${this.peerConnection.connectionState}`
        );
      });

      this.peerConnection.addEventListener("signalingstatechange", () => {
        console.log(
          `Signaling state change: ${this.peerConnection.signalingState}`
        );
      });

      this.peerConnection.addEventListener("iceconnectionstatechange ", () => {
        console.log(
          `ICE connection state change: ${this.peerConnection.iceConnectionState}`
        );
      });
    },
    startObjectDetection() {
      //create a canvas to grab an image for upload
      let imageCanvas = document.createElement("canvas");
      let imageCtx = imageCanvas.getContext("2d");

      const v = document.getElementById("remoteVideo");
      // const v = document.getElementById("localVideo");
      let videoHeight = 480;
      let videoWidth = 480;
      let uploadWidth = 480;

      imageCanvas.width = uploadWidth;
      imageCanvas.height = uploadWidth * (videoHeight / videoWidth);

      //Save and send the first image
      imageCtx.drawImage(
        v,
        0,
        0,
        videoWidth,
        videoHeight,
        0,
        0,
        uploadWidth,
        uploadWidth * (360 / videoWidth)
      );

      // document.body.appendChild(imageCanvas);
      imageCanvas.toBlob(this.postFile, "image/jpeg");
    },
    postFile(file) {
      // const server_url = "http://localhost:8000";
      const server_url = "https://j3b307.p.ssafy.io";
      //Set options as form data
      let formdata = new FormData();

      formdata.append("image", file);

      let url = `${server_url}/api/AI/emotions/test/`;
      axios
        .post(url, formdata)
        .then((res) => {
          if (res.data.length > 1) {
            this.expression = res.data[1];
          } else {
            this.expression = "question";
          }
          console.log(res.data);
        })
        .catch((err) => {
          console.log(err);
        });
    },
    captureAPI(operation) {
      if (operation) {
        this.repeatAPI = setInterval(() => {
          this.startObjectDetection();
        }, 2000);
      } else {
        clearInterval(this.repeatAPI);
        // setTimeout(() => {
        //   clearInterval(this.repeatAPI);
        // }, 20000);
      }
    },
    async friendReq() {
      const friendReqRef = await this.$firestore
        .collection("users")
        .doc(parseRoomLink(this.$route.params.id));
      await friendReqRef.update({
        friendsReq: this.$firebase.firestore.FieldValue.arrayUnion({
          uid: this.uid,
          nickname: this.nickname,
        }),
      });
    },
  },
  watch: {
    messages: function () {
      let usrIdx = this.messages.length - 1;
      this.other = this.messages[usrIdx];
      if (this.other["nickname"] != this.nickname) {
        const userRef = firebase.firestore().collection("users");
        userRef
          .where("nickname", "==", this.other["nickname"])
          .onSnapshot((snapshot) => {
            snapshot.forEach((res) => {
              this.otherGender = res.data()["gender"];
            });
          });
      }
    },
  },
};
</script>

<style>
/* https://www.w3schools.com/howto/howto_css_chat.asp */
.container {
  /* border: 2px solid #00d8d6; */
  /* background-color: #f1f1f1; */
  background-color: #f5fffa;
  border-radius: 5px;
  padding: 10px;
  margin: 10px 0;
}

/* mychat chat container */
.mychat {
  /* border-color: #78e08f; */
  /* background-color: #ddd; */
  background-color: #f0f8ff;
}

/* Clear floats */
.container::after {
  content: "";
  clear: both;
  display: table;
}

/* .container .username {
  float: left;
  max-width: 60px;
  width: 100%;
  margin-right: 20px;
  border-radius: 50%;
} */

.chat_right {
  /* float: right; */
  text-align: right;
  color: #aaa;
  margin-top: 0px;
  /* margin-right: 10px; */
  font-size: 0.75rem;
  /* border: 1px solid #aaa; */
  padding: 3px;
}

.chat_left {
  /* float: left; */
  text-align: left;
  color: #999;
  margin-top: 0px;
  /* margin-left: 10px; */
  font-size: 0.75rem;
  /* border: 1px solid #aaa; */
  padding: 3px;
}

.rel {
  position: relative;
}

.video_box {
  justify-self: center;
  position: absolute;
  left: 0;
  right: 0;
  top: 0;
  margin-left: auto;
  margin-right: auto;
  width: 320px;
  height: 240px;
}

.blank {
  height: 250px;
}

.self {
  text-align: left;
  margin-top: 3px;
  margin-left: 5px;
  font-size: 1rem;
}

.other {
  text-align: right;
  margin-top: 3px;
  margin-right: 5px;
  font-size: 1rem;
}

.home {
  position: absolute;
  top: 20px;
}
</style>
