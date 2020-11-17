<template>
  <div class="flex items-center justify-center h-screen bg-gray-200">
    <div class="flex flex-col w-full h-full max-w-lg shadow bg-yellow-100 p-4">
      <h2
        class="text-2xl font-semibold text-gray-900 text-center padding-top-10 mb-5"
      >
        회원가입
      </h2>
      <form>
        <div class="flex flex-wrap -mx-3 mb-6">
          <div class="w-full px-3 mb-6 md:mb-0">
            <label
              class="flex uppercase tracking-wide text-gray-700 text-xs font-bold mb-2"
            >
              별명
              <svg
                class="h-4 text-yellow-500 fill-current"
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  stroke-width="2"
                  d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z"
                />
              </svg>
            </label>
            <div class="flex">
              <input
                v-model="userData.nickname"
                class="appearance-none block w-3/4 bg-gray-100 text-gray-900 border border-gray-300 placeholder-gray-500 rounded-md py-3 px-4 mb-3 leading-tight focus:outline-none focus:border-blue-800"
                type="text"
                placeholder="2~8자리의 별명"
                maxlength="8"
              />
              <p v-if="canUseNickname" class="text-green-500 text-xs italic">
                사용 가능한 별명입니다
              </p>
              <div class="ml-2 w-1/4">
                <button
                  class="focus:outline-none bg-blue-800 h-12 w-24 text-white font-medium rounded-md items-center justify-center"
                  @click="checkNickname"
                >
                  별명 확인
                </button>
              </div>
            </div>
          </div>
          <div class="w-full px-3 mb-6 md:mb-0">
            <label
              class="flex uppercase tracking-wide text-gray-700 text-xs font-bold mb-2"
            >
              이메일
              <svg
                class="h-4 text-blue-500 stroke-current"
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
            </label>
            <input
              v-model="signupData.email"
              class="appearance-none block w-full bg-gray-100 text-gray-900 border border-gray-300 placeholder-gray-500 rounded-md py-3 px-4 mb-3 leading-tight focus:outline-none focus:bg-white focus:border-blue-800"
              type="text"
              placeholder="이메일을 입력해주세요"
              maxlength="30"
            />
            <!-- <p v-if="canUseEmail" class="text-green-500 text-xs italic">
              사용 가능한 이메일입니다
            </p> -->
          </div>
          <!-- <button @click="checkEmail">이메일 확인</button> -->
        </div>
        <div class="flex flex-wrap -mx-3 mb-6">
          <div class="w-full px-3">
            <label
              class="flex uppercase tracking-wide text-gray-700 text-xs font-bold mb-2"
            >
              비밀번호
              <svg
                class="h-4 text-red-500 stroke-current"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  stroke-width="2"
                  d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z"
                />
              </svg>
            </label>
            <input
              v-model="signupData.password"
              class="appearance-none block w-full bg-gray-100 text-gray-900 border border-gray-300 placeholder-gray-500 rounded-md py-3 px-4 mb-3 leading-tight focus:outline-none focus:bg-white focus:border-blue-800"
              type="password"
              placeholder="8~16자리의 비밀번호"
            />
            <!-- <p v-if="isPasswordShort" class="text-red-500 text-xs italic">
          비밀번호는 8자리 이상 16자리 이하입니다
        </p> -->
          </div>
          <div class="w-full px-3">
            <label
              class="flex uppercase tracking-wide text-gray-700 text-xs font-bold mb-2"
            >
              비밀번호 확인
              <svg
                class="h-4 text-red-500 fill-current"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  stroke-width="2"
                  d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z"
                />
              </svg>
            </label>
            <input
              v-model="signupData.passwordConfirm"
              class="appearance-none block w-full bg-gray-100 text-gray-900 border border-gray-300 placeholder-gray-500 rounded-md py-3 px-4 mb-3 leading-tight focus:outline-none focus:bg-white focus:border-blue-800"
              type="password"
              placeholder="비밀번호를 재입력해주세요"
            />
            <!-- <p v-if="!isPasswordSame" class="text-red-500 text-xs italic">
          비밀번호를 확인해주세요
        </p> -->
          </div>
        </div>
        <div class="flex flex-wrap -mx-3 mb-2">
          <div class="w-full md:w-1/3 px-3 mb-6 md:mb-0">
            <label
              class="flex uppercase tracking-wide text-gray-700 text-xs font-bold mb-2"
            >
              나이
              <svg
                class="h-4 text-yellow-900 fill-current"
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
            </label>
            <input
              v-model="userData.age"
              class="appearance-none block w-full bg-gray-100 text-gray-900 border border-gray-300 placeholder-gray-500 rounded-md py-3 px-4 leading-tight focus:outline-none focus:bg-white focus:border-blue-800"
              type="text"
              placeholder="양심ㅋ"
              maxlength="8"
            />
          </div>
          <div class="w-full md:w-1/3 px-3 mb-6 md:mb-0">
            <label
              class="flex uppercase tracking-wide text-gray-700 text-xs font-bold mb-2"
            >
              성별
              <svg
                class="h-4 text-gray-600"
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
            </label>
            <div class="relative">
              <select
                v-model="userData.gender"
                class="block appearance-none w-full bg-gray-100 text-gray-900 border border-gray-300 rounded-md py-3 px-4 pr-8 leading-tight focus:outline-none focus:bg-white focus:border-blue-800"
              >
                <option value="male">남</option>
                <option value="female">여</option>
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
          <div class="w-full md:w-1/3 px-3 mb-6 md:mb-0">
            <label
              class="flex uppercase tracking-wide text-gray-700 text-xs font-bold mb-2"
            >
              MBTI
              <svg
                class="h-4 text-teal-700"
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
            </label>
            <div class="relative">
              <select
                v-model="userData.mbti"
                class="block appearance-none w-full bg-gray-100 text-gray-900 border border-gray-300 rounded-md py-3 px-4 pr-8 leading-tight focus:outline-none focus:bg-white focus:border-blue-800"
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
        <div class="mt-5 flex items-center justify-center w-full">
          <button
            class="focus:outline-none flex items-center justify-center bg-blue-800 w-full text-white font-medium py-2 px-4 rounded-md"
            @click="DoSignup"
          >
            <span
              ><img class="h-5" src="../assets/ubgul.png" alt="ubgullogo"
            /></span>
            로 회원가입
          </button>
        </div>
      </form>
    </div>
  </div>
</template>

<script>
import firebase from "firebase";

export default {
  name: "Signup",
  data() {
    return {
      signupData: {
        email: "",
        password: "",
        passwordConfirm: "",
      },
      userData: {
        gender: "male",
        nickname: "",
        mbti: "none",
        age: "",
        friends: [],
      },
      canUseNickname: false,
    };
  },
  methods: {
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
    DoSignup(e) {
      e.preventDefault();
      if (this.userData.nickname == "") {
        alert("별명을 입력해주세요");
      } else if (!this.canUseNickname) {
        alert("사용 가능한 별명인지 확인해주세요");
      } else if (this.userData.age == "") {
        alert("나이를 입력해주세요");
      } else if (this.userData.age.length > 2 || isNaN(this.userData.age)) {
        alert("나이를 확인해주세요");
      } else if (this.signupData.password == "") {
        alert("비밀번호를 설정해주세요");
      } else if (this.signupData.password.length < 8) {
        alert("비밀번호는 8~16자리입니다");
      } else if (this.signupData.passwordConfirm == "") {
        alert("비밀번호 확인란을 입력해주세요");
      } else if (this.signupData.password !== this.signupData.passwordConfirm) {
        alert("비밀번호와 비밀번호 확인이 다릅니다");
      } else {
        firebase
          .auth()
          .createUserWithEmailAndPassword(
            this.signupData.email,
            this.signupData.password
          )
          .then((res) => {
            const userUid = res.user.uid;
            this.$store.commit("userUID", { uid: userUid });
            firebase
              .auth()
              .setPersistence(firebase.auth.Auth.Persistence.LOCAL);
            const db = firebase.firestore();
            db.collection("users").doc(userUid).set(this.userData);
            this.$store.commit("profile", this.userData);
            this.$router.push({ name: "Main" });
            console.log("로그인 성공");
          })
          .catch(function (error) {
            if (error.code == "auth/email-already-in-use") {
              alert("이미 사용중인 이메일입니다");
            } else {
              alert(error.message);
            }
          });
      }
    },
  },
};
</script>

<style>
.checkbutton {
  padding-right: 4px;
}
</style>
