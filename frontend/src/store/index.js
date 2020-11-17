import Vue from "vue";
import Vuex from "vuex";
import createPersistedState from "vuex-persistedstate";

Vue.use(Vuex);

const getDefaultState = () => {
  return {
    // user 정보
    uid: null,
    nickname: null,
    birthday: null,
    gender: null,
    mbti: null,
    friends: null,
    location: {
      lat: null,
      lon: null,
    },
    // signalling 관련 정보
    signalConf: {
      iceServers: [
        {
          urls: [
            "stun:stun1.l.google.com:19302",
            "stun:stun2.l.google.com:19302",
          ],
        },
        {
          urls: "turn:numb.viagenie.ca",
          username: "uutaein@gmail.com",
          password: "rlaxodls",
        },
      ],
      iceCandidatePoolSize: 10,
    },
    peerConnection: null,
  };
};
export default new Vuex.Store({
  state: getDefaultState(),
  plugins: [createPersistedState()],
  mutations: {
    userUID(state, payload) {
      state.uid = payload.uid;
    },
    Logout(state) {
      Object.assign(state, getDefaultState());
    },
    profile(state, payload) {
      state.nickname = payload.nickname;
      state.birthday = payload.birthday;
      state.gender = payload.gender;
      state.mbti = payload.mbti;
      state.friends = payload.friends;
      state.age = payload.age;
    },
    peerConnection(state, payload) {
      state.peerConnection = payload.pc;
    },
    location(state, payload) {
      state.location.lat = payload.lat;
      state.location.lon = payload.lat;
    },
  },
  actions: {},
  modules: {},
});
