import Vue from 'vue'
import App from './App.vue'
import './registerServiceWorker'
import router from './router'
import store from './store'
import './assets/tailwind.css'

// form validation
import VueFormulate from '@braid/vue-formulate'
Vue.use(VueFormulate, {
  classes: {
    label: 'font-medium text-sm',
    help: 'text-xs mb-1 text-gray-600',
    error: 'text-red-700 text-xs mb-1'
  }
})
// webrtc polyfill
// eslint-disable-next-line no-unused-vars
import adapter from 'webrtc-adapter';


//firebase 관련 설정
import firebase from "firebase";
var firebaseConfig = {
  apiKey: "AIzaSyBtxpj_R-CEUjwSTuFJ11jQ75qlJh5mKQE",
  authDomain: "up-gul.firebaseapp.com",
  databaseURL: "https://up-gul.firebaseio.com",
  projectId: "up-gul",
  storageBucket: "up-gul.appspot.com",
  messagingSenderId: "670567766799",
  appId: "1:670567766799:web:7b31c3549169294decfc7a",
  measurementId: "G-VWCDMT6Y5Y"
};
firebase.initializeApp(firebaseConfig);
firebase.analytics();
const firestore = firebase.firestore();
const realtimedb = firebase.database();

Vue.prototype.$firebase = firebase;
Vue.prototype.$firestore = firestore;
Vue.prototype.$db = realtimedb;

// vue chat scroll
import VueChatScroll from 'vue-chat-scroll'
Vue.use(VueChatScroll)

Vue.config.productionTip = false

new Vue({
  router,
  store,
  render: h => h(App)
}).$mount('#app')
