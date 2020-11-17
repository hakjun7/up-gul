<template>
  <div class="h-screen flex items-center justify-center bg-gray-200">
    <div
      class="flex flex-col w-full h-full max-w-lg shadow bg-yellow-100 px-4 pt-16"
    >
      <!-- <div class="max-w-md w-full"> -->
      <div>
        <img class="mx-auto h-16" src="../assets/ubgul.png" alt="ubgullogo" />
      </div>
      <form class="mt-8" action="#" method="POST">
        <input type="hidden" name="remember" value="true" />

        <div>
          <input
            v-model="loginData.email"
            aria-label="Email address"
            name="email"
            type="email"
            required
            class="bg-gray-100 mb-3 appearance-none rounded-md relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 focus:outline-none focus:bg-white focus:border-blue-800 focus:z-10 sm:text-sm sm:leading-5"
            placeholder="이메일"
          />
        </div>
        <div class="-mt-px">
          <input
            v-model="loginData.password"
            aria-label="Password"
            name="password"
            type="password"
            required
            class="bg-gray-100 appearance-none rounded-md relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 focus:outline-none focus:bg-white focus:border-blue-800 focus:z-10 sm:text-sm sm:leading-5"
            placeholder="비밀번호"
          />
        </div>

        <div class="mt-3 flex items-center justify-between">
          <div class="flex items-center text-sm">
            <span
              @click="goSignup"
              class="changeCursor flex font-medium text-gray-600 hover:text-blue-800 focus:outline-none focus:underline transition ease-in-out duration-150"
            >
              회원가입
              <!-- <img
                class="ml-1 h-5 w-5"
                src="../assets/ubgul.png"
                alt="ubgullogo"
              /> -->
            </span>
          </div>

          <div class="text-sm leading-5">
            <span
              @click="openModal"
              class="changeCursor font-medium text-gray-600 hover:text-blue-800 focus:outline-none focus:underline transition ease-in-out duration-150"
            >
              비밀번호를 잊어버리셨나요?
            </span>
          </div>
        </div>

        <div class="mt-8">
          <button
            @click="upgulLogin"
            type="submit"
            class="bg-blue-800 h-12 group relative w-full flex items-center justify-center py-2 px-4 border border-transparent leading-5 font-medium rounded-md text-white focus:outline-none focus:border-indigo-700 focus:shadow-outline-indigo active:bg-indigo-700 transition duration-150 ease-in-out"
          >
            로그인
          </button>
          <a
            href="#"
            class="h-12 flex items-center justify-center mt-3 text-white border border-blue-800 rounded-md bg-gray-100"
          >
            <div class="px-4 py-3">
              <svg class="h-6 w-6" viewBox="0 0 40 40">
                <path
                  d="M36.3425 16.7358H35V16.6667H20V23.3333H29.4192C28.045 27.2142 24.3525 30 20 30C14.4775 30 10 25.5225 10 20C10 14.4775 14.4775 9.99999 20 9.99999C22.5492 9.99999 24.8683 10.9617 26.6342 12.5325L31.3483 7.81833C28.3717 5.04416 24.39 3.33333 20 3.33333C10.7958 3.33333 3.33335 10.7958 3.33335 20C3.33335 29.2042 10.7958 36.6667 20 36.6667C29.2042 36.6667 36.6667 29.2042 36.6667 20C36.6667 18.8825 36.5517 17.7917 36.3425 16.7358Z"
                  fill="#FFC107"
                />
                <path
                  d="M5.25497 12.2425L10.7308 16.2583C12.2125 12.59 15.8008 9.99999 20 9.99999C22.5491 9.99999 24.8683 10.9617 26.6341 12.5325L31.3483 7.81833C28.3716 5.04416 24.39 3.33333 20 3.33333C13.5983 3.33333 8.04663 6.94749 5.25497 12.2425Z"
                  fill="#FF3D00"
                />
                <path
                  d="M20 36.6667C24.305 36.6667 28.2167 35.0192 31.1742 32.34L26.0159 27.975C24.3425 29.2425 22.2625 30 20 30C15.665 30 11.9842 27.2359 10.5975 23.3784L5.16254 27.5659C7.92087 32.9634 13.5225 36.6667 20 36.6667Z"
                  fill="#4CAF50"
                />
                <path
                  d="M36.3425 16.7358H35V16.6667H20V23.3333H29.4192C28.7592 25.1975 27.56 26.805 26.0133 27.9758C26.0142 27.975 26.015 27.975 26.0158 27.9742L31.1742 32.3392C30.8092 32.6708 36.6667 28.3333 36.6667 20C36.6667 18.8825 36.5517 17.7917 36.3425 16.7358Z"
                  fill="#1976D2"
                />
              </svg>
            </div>
            <span
              class="px-4 py-3 w-5/6 text-center text-blue-800 font-medium"
              @click="googleLogin"
            >
              구글 계정으로 로그인
            </span>
          </a>
        </div>
      </form>
    </div>
    <div
      v-if="showDialog"
      class="fixed pin overflow-auto w-full h-full bg-smoke-light z-50 flex"
    >
      <div
        class="rounded relative p-8 bg-white w-full max-w-md m-auto flex-col flex"
      >
        <h2 class="text-2xl font-semibold text-gray-700 text-center mb-5">
          추가 정보를 입력해 주세요
        </h2>
        <label class="block">
          <span class="text-gray-700">별명</span>
          <input
            class="form-input mt-1 block w-full"
            v-model="userData.nickname"
            placeholder="2~8자리의 별명"
            maxlength="8"
          />
          <button @click="checkNickname">별명 확인</button>
        </label>
        <div class="mt-4">
          <span class="text-gray-700">성별</span>
          <div class="mt-2">
            <label class="inline-flex items-center">
              <input
                type="radio"
                class="form-radio"
                name="gender"
                v-model="userData.gender"
                value="male"
              />
              <span class="ml-2">남자</span>
            </label>
            <label class="inline-flex items-center ml-6">
              <input
                type="radio"
                class="form-radio"
                name="gender"
                v-model="userData.gender"
                value="female"
              />
              <span class="ml-2">여자</span>
            </label>
          </div>
        </div>
        <label class="block mt-4">
          <span class="text-gray-700">나이</span>
          <input
            class="form-input mt-1 w-full"
            type="text"
            v-model="userData.age"
            placeholder="나이"
            maxlength="2"
          />
        </label>
        <label class="block mt-4">
          <span class="text-gray-700">MBTI</span>
          <select v-model="userData.mbti" class="form-select mt-1 block w-full">
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
        </label>
        <button
          @click="submitProfile"
          class="bg-blue-500 hover:bg-blue-700 mt-5 text-white font-bold py-2 px-4 border border-blue-700 rounded"
        >
          제출하기
        </button>
      </div>
    </div>
    <div
      v-if="openPasswordModal"
      class="fixed pin overflow-auto w-full h-full bg-smoke-light z-50 flex"
    >
      <div
        class="rounded relative p-8 bg-yellow-100 w-full max-w-md m-auto flex-col flex"
      >
        <h2 class="text-2xl font-semibold text-gray-900 text-center mb-5">
          비밀번호 찾기
        </h2>
        <svg
          class="w-8 h-8 cursor-pointer absolute right-0 top-0 mt-3 mr-3 text-blue-800"
          @click="closeModal"
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
        <label class="block mt-4">
          <span class="text-gray-700">이메일</span>
          <input
            class="h-10 px-3 py-2 mt-1 w-full focus:border-blue-800 outline-none placeholder-gray-500 bg-gray-100 border border-gray-300 rounded-md"
            type="text"
            v-model="changePassEmail"
            placeholder="이메일을 입력해주세요"
          />
        </label>
        <button
          @click="changePassword"
          class="bg-blue-800 mt-5 text-white font-medium py-2 px-4 border border-blue-800 rounded-md focus:outline-none"
        >
          변경하기
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { mapState } from "vuex";
import firebase from "firebase";
const provider = new firebase.auth.GoogleAuthProvider();
provider.setCustomParameters({ display: "popup" });

export default {
  name: "Login",
  computed: mapState(["uid"]),
  data() {
    return {
      loginData: {
        email: "",
        password: "",
      },
      changePassEmail: "",
      errorMsg: null,
      showDialog: false,
      openPasswordModal: false,
      canUseNickname: false,
      userData: {
        gender: "male",
        nickname: "",
        mbti: "none",
        age: "",
        friends: [],
      },
    };
  },
  methods: {
    openModal() {
      this.openPasswordModal = true;
    },
    closeModal() {
      this.openPasswordModal = false;
    },
    changePassword() {
      if (this.changePassEmail !== "") {
        console.log(this.changePassEmail);
        firebase
          .auth()
          .sendPasswordResetEmail(this.changePassEmail)
          .then(() => {
            alert("비밀번호 변경 요청 완료!");
            this.changePassEmail = "";
            this.closeModal();
          })
          .catch((err) => {
            alert(err);
          });
      } else {
        alert("이메일을 입력해주세요");
      }
    },
    async checkNickname(e) {
      e.preventDefault();
      if (this.userData.nickname.length < 2) {
        this.canUseNickname = false;
        alert("별명은 2~8자리여야 합니다");
      } else {
        let db = firebase.firestore();
        const userRef = db.collection("users");
        const snapshot = await userRef
          .where("nickname", "==", this.userData.nickname)
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
    async googleLogin(e) {
      e.preventDefault();
      try {
        const res = await firebase.auth().signInWithPopup(provider);
        const user = res.user.uid;
        // uid 를 vuex 에 저장
        this.$store.commit("userUID", { uid: user });
        // 인증 상태 지속 유형 변경
        firebase.auth().setPersistence(firebase.auth.Auth.Persistence.LOCAL);
        // realtime db 에 유저정보 있으면 vuex 에 유저정보 저장, 아니면 강제로 profile로 redirect
        const db = firebase.firestore();
        const userDoc = db.collection("users").doc(user);
        const doc = await userDoc.get();
        if (doc.exists) {
          this.$store.commit("profile", doc.data());
          // vuex 에 문서 내용을 저장하고 Main 으로 Push
          this.$router.push({ name: "Main" });
        } else {
          // 추가적인 정보 입력 모달을 띄우고, 모달 입력 정보를 저장하자.
          this.showDialog = true;
          console.log("모달 작업중");
        }
      } catch (err) {
        console.log(err);
      }
    },
    async submitProfile() {
      if (this.userData.nickname === "") {
        alert("별명을 정해주세요");
      } else if (this.userData.age.length > 2 || isNaN(this.userData.age)) {
        alert("나이를 확인해주세요");
      } else {
        if (!this.canUseNickname) {
          alert("별명을 인증받아주세요");
        } else {
          try {
            const db = firebase.firestore();
            await db.collection("users").doc(this.uid).set(this.userData);
            this.$store.commit("profile", this.userData);
            this.$router.push({ name: "Main" });
          } catch (e) {
            console.log(e);
          }
        }
      }
    },
    goSignup() {
      this.$router.push({ name: "Signup" });
    },
    upgulLogin(e) {
      e.preventDefault();
      firebase
        .auth()
        .signInWithEmailAndPassword(
          this.loginData.email,
          this.loginData.password
        )
        .then((res) => {
          const userUid = res.user.uid;
          this.$store.commit("userUID", { uid: userUid });
          firebase.auth().setPersistence(firebase.auth.Auth.Persistence.LOCAL);
          this.$router.push("/");
          const db = firebase.firestore();
          db.collection("users")
            .doc(userUid)
            .get()
            .then((reply) => {
              this.$store.commit("profile", reply.data());
              this.$router.push({ name: "Main" });
            })
            .catch(() => {
              alert("문제발생!!!");
            });
        })
        .catch((error) => {
          const errorCode = error.code;
          if (errorCode === "auth/wrong-password") {
            alert("비밀번호가 잘못되었습니다");
          } else if (errorCode === "auth/user-not-found") {
            alert("존재하지 않는 유저입니다");
          } else if (errorCode === "auth/invalid-email") {
            alert("유효하지 않은 이메일입니다");
          } else {
            alert(error.message);
          }
        });
    },
  },
};
</script>

<style>
.changeCursor {
  cursor: pointer;
}
</style>
