<template>
  <div class="flex items-center justify-center h-screen bg-gray-200">
    <div class="flex flex-col w-full h-full max-w-lg shadow bg-yellow-100 p-4">
      <h2 class="flex flex-row items-center">
        <div class="mr-auto flex flex-row items-center justify-between mb-3">
          <img class="h-12" src="../assets/ubgul.png" alt="" />
        </div>
        <button
          @click="logOut"
          class="focus:outline-none relative inline-block text-blue-800 stroke-current"
        >
          <svg
            class="h-8"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1"
            />
          </svg>
        </button>
      </h2>
      <div
        class="cursor-default mt-3 px-3 flex rounded h-16 w-full items-center bg-gradient-to-r from-blue-800 to-yellow-400"
      >
        <span class="mx-auto font-sans font-black text-white text-lg"
          >My Page</span
        >
      </div>
      <div class="flex flex-row items-center justify-center mt-10 mb-10">
        <div class="relative flex-shrink-0">
          <button
            @click="showModal = true"
            class="focus:outline-none stroke-current current-2 text-gray-700 hover:text-blue-800 absolute right-0 top-0 mt-px mr-px flex items-center justify-center"
          >
            <svg class="h-6 w-6" fill="currentColor" viewBox="0 0 20 20">
              <path
                fill-rule="evenodd"
                d="M11.49 3.17c-.38-1.56-2.6-1.56-2.98 0a1.532 1.532 0 01-2.286.948c-1.372-.836-2.942.734-2.106 2.106.54.886.061 2.042-.947 2.287-1.561.379-1.561 2.6 0 2.978a1.532 1.532 0 01.947 2.287c-.836 1.372.734 2.942 2.106 2.106a1.532 1.532 0 012.287.947c.379 1.561 2.6 1.561 2.978 0a1.533 1.533 0 012.287-.947c1.372.836 2.942-.734 2.106-2.106a1.533 1.533 0 01.947-2.287c1.561-.379 1.561-2.6 0-2.978a1.532 1.532 0 01-.947-2.287c.836-1.372-.734-2.942-2.106-2.106a1.532 1.532 0 01-2.287-.947zM10 13a3 3 0 100-6 3 3 0 000 6z"
                clip-rule="evenodd"
              />
            </svg>
          </button>
          <div
            class="text-white font-bold bg-gradient-to-bl from-yellow-500 via-blue-800 to-yellow-300 font-mono text-3xl flex rounded-full h-32 w-32 items-center justify-center"
          >
            {{ nickname }}
          </div>
        </div>
      </div>
      <div class="py-4 px-4" style="height: 400px">
        <div class="flex items-center mt-4 text-gray-700 font-bold">
          <svg
            class="h-6 text-blue-500 stroke-current"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
            />
          </svg>
          <p class="px-1 text-base">
            이메일 <span class="ml-3">{{ email }}</span>
          </p>
        </div>
        <div class="flex items-center mt-4 text-gray-700 font-bold">
          <svg
            class="h-6 text-gray-600"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M5.121 17.804A13.937 13.937 0 0112 16c2.5 0 4.847.655 6.879 1.804M15 10a3 3 0 11-6 0 3 3 0 016 0zm6 2a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
          <p class="px-1 text-base">
            성별 <span class="ml-6">{{ gender }}</span>
          </p>
        </div>
        <div class="flex items-center mt-4 text-gray-700 font-bold">
          <svg
            class="h-6 text-yellow-900 fill-current"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M3 6l3 1m0 0l-3 9a5.002 5.002 0 006.001 0M6 7l3 9M6 7l6-2m6 2l3-1m-3 1l-3 9a5.002 5.002 0 006.001 0M18 7l3 9m-3-9l-6-2m0-2v2m0 16V5m0 16H9m3 0h3"
            />
          </svg>
          <p class="px-1 text-base">
            나이 <span class="ml-6">{{ age }}</span>
          </p>
        </div>
        <div class="flex items-center mt-4 text-gray-700 font-bold">
          <svg
            class="h-6 text-teal-700"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M16 8v8m-4-5v5m-4-2v2m-2 4h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
            />
          </svg>
          <p class="px-1 text-base">
            MBTI <span class="ml-4">{{ mbti }}</span>
          </p>
        </div>
        <div class="flex items-center mt-4 text-gray-700 font-bold">
          <svg
            class="h-6 text-blue-800"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z"
            />
          </svg>
          <div class="flex px-1 text-base">
            친구
            <div class="ml-4">
              <div
                class="relative inline-block"
                @click="isVisible = !isVisible"
              >
                <button
                  type="button"
                  class="inline-flex items-center justify-between px-2 py-1 text-blue-800 transition-all duration-500 focus:outline-none"
                >
                  <svg
                    class="h-4"
                    xmlns="http://www.w3.org/2000/svg"
                    viewBox="0 0 20 20"
                    fill="currentColor"
                  >
                    <path
                      d="M10 2a6 6 0 00-6 6v3.586l-.707.707A1 1 0 004 14h12a1 1 0 00.707-1.707L16 11.586V8a6 6 0 00-6-6zM10 18a3 3 0 01-3-3h6a3 3 0 01-3 3z"
                    />
                  </svg>
                </button>
                <transition
                  enter-active-class="transition duration-300 ease-out transform"
                  enter-class="-translate-y-3 scale-95 opacity-0"
                  enter-to-class="translate-y-0 scale-100 opacity-100"
                  leave-active-class="transition duration-150 ease-in transform"
                  leave-class="translate-y-0 opacity-100"
                  leave-to-class="-translate-y-3 opacity-0"
                >
                  <div v-show="isVisible" class="absolute bottom-0 pt-2 z-10">
                    <div
                      class="ml-12 relative py-1 bg-blue-100 border border-gray-200 rounded-md w-3/4"
                    >
                      <ul class="relative">
                        <p
                          class="block w-full px-4 py-2 font-medium text-gray-700 whitespace-no-wrap focus:outline-none transition duration-300 ease-in-out"
                        >
                          친구 목록
                        </p>
                        <p
                          v-if="friends.length === 0"
                          class="block w-full px-4 py-2 font-medium text-red-700 whitespace-no-wrap focus:outline-none transition duration-300 ease-in-out"
                        >
                          친구가 없습니다
                        </p>
                        <li
                          v-for="(friend, idx) in friends"
                          :key="idx"
                          class="block w-full px-4 py-2 font-medium text-gray-700 whitespace-no-wrap hover:bg-gray-100 focus:outline-none hover:text-gray-900 focus:text-gray-900 focus:shadow-outline transition duration-300 ease-in-out"
                        >
                          {{ friend.nickname }}
                        </li>
                      </ul>
                    </div>
                  </div>
                </transition>
              </div>
            </div>
          </div>
        </div>
        <transition name="fade" appear>
          <div
            class="modal-overlay"
            v-if="showModal"
            @click="showModal = false"
          ></div>
        </transition>
        <transition appear>
          <div class="modal bg-yellow-100" v-if="showModal">
            <h2 class="text-2xl font-semibold text-gray-900 text-center mb-5">
              정보 변경
            </h2>
            <svg
              class="w-8 h-8 cursor-pointer absolute right-0 top-0 mt-3 mr-3 text-blue-800"
              @click="showModal = false"
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 20 20"
              fill="currentColor"
            >
              <path
                fill-rule="evenodd"
                d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                clip-rule="evenodd"
              />
            </svg>
            <div class="flex flex-wrap w-full -mx-3 mb-6">
              <div class="w-3/4 px-3 mb-6 md:mb-0">
                <label
                  class="block uppercase tracking-wide text-gray-700 text-xs font-bold mb-2"
                >
                  별명
                </label>
                <input
                  v-model="changeNick"
                  class="appearance-none block w-full bg-gray-200 text-gray-700 border border-gray-200 rounded py-3 px-4 mb-3 leading-tight focus:outline-none focus:bg-white focus:border-gray-500"
                  type="text"
                  placeholder="2~8자리의 별명"
                  maxlength="8"
                />
                <p v-if="canUseNickname" class="text-green-500 text-xs italic">
                  사용 가능한 별명입니다
                </p>
              </div>
              <div class="w-1/4 justify-center align-items pt-6">
                <button
                  class="focus:outline-none bg-blue-800 h-12 w-24 text-white font-medium rounded-md items-center justify-center"
                  @click="checkNickname"
                >
                  중복 확인
                </button>
              </div>
              <div class="w-full mb-6 md:mb-0">
                <div class="w-full md:w-2/3 px-3 mb-6 md:mb-0">
                  <label
                    class="block uppercase tracking-wide text-gray-700 text-xs font-bold mb-2"
                  >
                    MBTI
                  </label>
                  <div class="relative">
                    <select
                      v-model="mbti"
                      class="block appearance-none w-full bg-gray-200 border border-gray-200 text-gray-700 py-3 px-4 pr-8 rounded leading-tight focus:outline-none focus:bg-white focus:border-gray-500"
                    >
                      <option value="none">없음</option>
                      <option value="ISTJ">ISTJ - 소금형</option>
                      <option value="ISFJ">ISFJ - 권력형</option>
                      <option value="ESTP">ESTP - 활동가형</option>
                      <option value="ESFP">ESFP - 사교형</option>
                      <option value="INFJ">INFJ - 예언자형</option>
                      <option value="INTJ">INTJ - 과학자형</option>
                      <option value="ENFP">ENFP - 스파크형</option>
                      <option value="ENTP">ENTP - 발명가형</option>
                      <option value="ISTP">ISTP - 백과사전형</option>
                      <option value="ISFP">ISFP - 성인군자형</option>
                      <option value="ESTJ">ESTJ - 사업가형</option>
                      <option value="ESFJ">ESFJ - 친선도모형</option>
                      <option value="INFP">INFP - 잔다르크형</option>
                      <option value="INTP">INTP - 아이디어형</option>
                      <option value="ENFJ">ENFJ - 언변능숙형</option>
                      <option value="ENTJ">ENTJ - 지도자형</option>
                    </select>
                    <div
                      class="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-700"
                    >
                      <svg
                        class="fill-current h-4 w-4"
                        xmlns="http://www.w3.org/2000/svg"
                        viewBox="0 0 20 20"
                      >
                        <path
                          d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z"
                        />
                      </svg>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            <button
              class="bg-blue-800 w-full text-white font-bold py-2 px-4 rounded"
              @click="changeInfo"
            >
              변경하기
            </button>
          </div>
        </transition>
      </div>
      <div
        class="flex flex-row items-center justify-around mt-16 h-20 bg-blue-800"
      >
        <button
          @click="goToProfile"
          class="cursor-pointer flex text-yellow-500 stroke-current focus:outline-none"
        >
          <svg class="h-8 w-8" fill="currentColor" viewBox="0 0 20 20">
            <path d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z" />
          </svg>
        </button>
        <button
          @click="goToHome"
          class="cursor-pointer flex text-yellow-500 fill-current focus:outline-none"
        >
          <svg class="h-8 w-8" viewBox="0 0 20 20" fill="currentColor">
            <path
              d="M10.707 2.293a1 1 0 00-1.414 0l-7 7a1 1 0 001.414 1.414L4 10.414V17a1 1 0 001 1h2a1 1 0 001-1v-2a1 1 0 011-1h2a1 1 0 011 1v2a1 1 0 001 1h2a1 1 0 001-1v-6.586l.293.293a1 1 0 001.414-1.414l-7-7z"
            />
          </svg>
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import firebase from "firebase";

export default {
  created() {
    // this.online(),
    this.getfriends();
  },
  data() {
    return {
      showModal: false,
      uid: "",
      nickname: "",
      changeNick: "",
      canUseNickname: false,
      email: "",
      friends: [],
      mbti: "",
      age: "",
      gender: "",
      isVisible: false,
    };
  },
  methods: {
    goToProfile() {
      this.$router.push({ name: "Profile" });
    },
    goToHome() {
      this.$router.push({ name: "Main" });
    },
    logOut() {
      firebase
        .auth()
        .signOut()
        .then(() => {
          this.$router.push({ name: "Login" });
        });
    },
    async checkNickname() {
      if (this.changeNick.length < 2) {
        this.canUseNickname = false;
        alert("별명은 2~8자리여야 합니다");
      } else {
        let db = firebase.firestore();
        const userRef = db.collection("users");
        const snapshot = await userRef
          .where("nickname", "==", this.changeNick)
          .get();
        if (snapshot.empty) {
          this.canUseNickname = true;
          alert("사용 가능한 별명입니다");
        } else {
          this.canUseNickname = false;
          alert("이미 사용 중인 별명입니다");
        }
      }
    },
    async changeNickname() {
      const db = firebase.firestore();
      db.collection("users")
        .doc(this.uid)
        .update({ nickname: this.changeNick })
        .then(() => {
          const hello = this.$store.state;
          this.$store.commit("profile", {
            ...hello,
            nickname: this.changeNick,
          });
          this.nickname = this.changeNick;
        })
        .catch(() => {
          alert("별명 변경 실패!");
        });
    },
    async changeMBTI() {
      const db = firebase.firestore();
      db.collection("users")
        .doc(this.uid)
        .update({ mbti: this.changeMb })
        .then(() => {
          const hello = this.$store.state;
          this.$store.commit("profile", { ...hello, mbti: this.changeMb });
          this.mbti = this.changeMb;
        })
        .catch(() => {
          alert("MBTI 변경 실패!");
        });
    },
    async changeInfo() {
      if (this.changeNick == this.nickname) {
        this.changeMBTI();
        alert("정보 변경 성공");
      } else if (!this.canUseNickname) {
        alert("별명을 인증받아 주세요");
      } else {
        this.changeNickname();
        this.changeMBTI();
        alert("정보 변경 성공");
      }
    },
    findGender(v) {
      if (v == "male") {
        return "남성";
      }
      return "여성";
    },
    // async getFriends(uid) {
    //   console.log(uid);
    //   const db = firebase.firestore();
    //   var friendsDoc = await db.collection("users").doc(uid).get();
    //   return friendsDoc.data().friends;
    // },
    async getfriends() {
      const friendsRef = await this.$firestore
        .collection("users")
        .doc(this.uid);
      friendsRef.onSnapshot((snapshot) => {
        const friendsdata = snapshot.data()["friends"];
        if (friendsdata) {
          friendsdata.forEach((fri) => {
            if (!this.friends.includes(fri)) {
              this.friends.push(fri);
            }
          });
        }
      });
    },
    async getUserInfo() {
      this.uid = this.$store.state.uid;
      this.nickname = this.$store.state.nickname;
      this.changeNick = this.$store.state.nickname;
      this.mbti = this.$store.state.mbti;
      this.age = this.$store.state.age ? this.$store.state.age : "?!";
      this.gender = this.findGender(this.$store.state.gender);
      this.email = this.$firebase.auth().currentUser.email
        ? this.$firebase.auth().currentUser.email
        : this.$store.state.email;
      // this.friends = await this.getFriends(this.uid);
      // console.log(this.friends)
      // var user = await firebase.auth().currentUser;
      // if (user != null) {
      //     this.email = user.email;
      // }
    },
  },
  mounted() {
    this.getUserInfo();
  },
};
</script>



<style>
.modal-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  z-index: 98;
  background-color: rgba(0, 0, 0, 0.3);
}

.modal {
  position: fixed;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  z-index: 99;

  width: 100%;
  max-width: 400px;
  background-color: #fff;
  border-radius: 16px;

  padding: 25px;
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.5s;
}

.fade-enter,
.fade-leave-to {
  opacity: 0;
}
</style>