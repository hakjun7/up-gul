<template>
  <!-- component -->
  <div class="flex items-center justify-center h-screen bg-gray-200">
    <div class="flex flex-col w-full h-full max-w-lg shadow bg-yellow-100 p-4">
      <h2 class="flex flex-row items-center">
        <div class="mr-auto flex flex-row items-center justify-between mb-3">
          <img class="h-12" src="../assets/ubgul.png" alt="" />
        </div>
        <div class="relative inline-block" @click="isVisible = !isVisible">
          <button
            type="button"
            class="inline-flex items-center justify-between px-2 py-1 text-blue-800 transition-all duration-500 focus:outline-none"
          >
            <svg
              v-if="friendReqList.length === 0"
              class="h-8"
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 20 20"
              fill="currentColor"
            >
              <path
                d="M10 2a6 6 0 00-6 6v3.586l-.707.707A1 1 0 004 14h12a1 1 0 00.707-1.707L16 11.586V8a6 6 0 00-6-6zM10 18a3 3 0 01-3-3h6a3 3 0 01-3 3z"
              />
            </svg>
            <svg
              v-if="friendReqList.length !== 0"
              class="h-8 animate-bounce"
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
            <div v-show="isVisible" class="absolute right-0 pt-2 z-10">
              <div
                class="relative py-1 bg-blue-100 border border-gray-200 rounded-md shadow-xl"
              >
                <!-- <ul class="relative border-b-2">
                  <p
                    class="block w-full px-4 py-2 font-medium text-gray-700 whitespace-no-wrap  focus:outline-none transition duration-300 ease-in-out"
                  >
                    친구 목록
                  </p>
                  <p
                    v-if="friendsList.length === 0"
                    class="block w-full px-4 py-2 font-medium text-red-700 whitespace-no-wrap  focus:outline-none transition duration-300 ease-in-out"
                  >
                    친구가 없습니다
                  </p>
                  <li
                    v-for="(friend, idx) in friendsList"
                    :key="idx"
                    class="block w-full px-4 py-2 font-medium text-gray-700 whitespace-no-wrap hover:bg-gray-100 focus:outline-none hover:text-gray-900 focus:text-gray-900 focus:shadow-outline transition duration-300 ease-in-out"
                  >
                    {{ friend.nickname }}
                  </li>
                </ul> -->
                <ul class="relative">
                  <p
                    class="block w-full px-4 py-2 font-medium text-gray-900 whitespace-no-wrap focus:outline-none transition duration-300 ease-in-out"
                  >
                    친구요청 목록
                  </p>
                  <p
                    v-if="friendReqList.length === 0"
                    class="block w-full px-4 py-2 font-medium text-red-700 whitespace-no-wrap focus:outline-none transition duration-300 ease-in-out"
                  >
                    친구요청이 없습니다
                  </p>
                  <li
                    v-for="(friendCand, idx) in friendReqList"
                    :key="idx"
                    class="block w-full px-4 py-2 font-medium text-gray-700 whitespace-no-wrap hover:bg-gray-100 focus:outline-none hover:text-gray-900 focus:text-gray-900 focus:shadow-outline transition duration-300 ease-in-out"
                  >
                    <div
                      class="flex items-center justify-center mr-2 font-medium py-1 px-2 rounded-full"
                    >
                      <span class="mr-5"> {{ friendCand.nickname }}</span>
                      <button
                        @click="acceptFriend(friendCand, idx)"
                        class="flex items-center justify-center text-sm h-3 max-w-full mr-2 p-3 font-normal leading-none flex-initial rounded text-blue-700 bg-blue-100 border border-blue-300"
                      >
                        추가
                      </button>
                      <button
                        @click="rejectFriend(friendCand, idx)"
                        class="flex items-center justify-center text-sm h-3 max-w-full mr-2 p-3 font-normal leading-none flex-initial rounded text-pink-700 bg-pink-100 border border-pink-300"
                      >
                        거부
                      </button>
                    </div>
                  </li>
                </ul>
              </div>
            </div>
          </transition>
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

        <!--              <div class="relative inline-block" @mouseover="isVisible = true" @mouseleave="isVisible = false" @keydown.enter="isVisible = !isVisible">-->
        <!--                <button type="button" class="inline-flex items-center justify-between px-2 py-1 font-medium text-gray-700 transition-all duration-500 rounded-md focus:outline-none focus:text-brand-900 sm:focus:shadow-outline">-->
        <!--                  <span class="flex-shrink-0">알림</span>-->
        <!--                  <svg fill="currentColor" viewBox="0 0 20 20" class="flex-shrink-0 w-5 h-5 ml-1">-->
        <!--                    <path :class="{ 'rotate-180': isVisible }" class="transition duration-300 ease-in-out origin-center transform" fill-rule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clip-rule="evenodd"></path>-->
        <!--                  </svg>-->
        <!--                </button>-->
        <!--                <transition enter-active-class="transition duration-300 ease-out transform" enter-class="-translate-y-3 scale-95 opacity-0" enter-to-class="translate-y-0 scale-100 opacity-100" leave-active-class="transition duration-150 ease-in transform" leave-class="translate-y-0 opacity-100" leave-to-class="-translate-y-3 opacity-0">-->
        <!--                  <div v-show="isVisible" class="absolute right-0 pt-2 z-10">-->
        <!--                    <div class="relative py-1 bg-white border border-gray-200 rounded-md shadow-xl">-->
        <!--                      <div class="relative">-->
        <!--                        <a href="#" class="block w-full px-4 py-2 font-medium text-gray-700 whitespace-no-wrap hover:bg-gray-100 focus:outline-none hover:text-gray-900 focus:text-gray-900 focus:shadow-outline transition duration-300 ease-in-out">Submenu Link #1</a>-->
        <!--                        <a href="#" class="block w-full px-4 py-2 font-medium text-gray-700 whitespace-no-wrap hover:bg-gray-100 focus:outline-none hover:text-gray-900 focus:text-gray-900 focus:shadow-outline transition duration-300 ease-in-out">Submenu Link #2</a>-->
        <!--                        <a href="#" class="block w-full px-4 py-2 font-medium text-gray-700 whitespace-no-wrap hover:bg-gray-100 focus:outline-none hover:text-gray-900 focus:text-gray-900 focus:shadow-outline transition duration-300 ease-in-out">Submenu Link #3</a>-->
        <!--                        <a href="#" class="block w-full px-4 py-2 font-medium text-gray-700 whitespace-no-wrap hover:bg-gray-100 focus:outline-none hover:text-gray-900 focus:text-gray-900 focus:shadow-outline transition duration-300 ease-in-out">Submenu Link #4</a>-->
        <!--                      </div>-->
        <!--                    </div>-->
        <!--                  </div>-->
        <!--                </transition>-->
        <!--              </div>-->
      </h2>
      <div
        class="cursor-default mt-3 px-3 flex rounded h-12 w-full items-center bg-gradient-to-r from-yellow-400 to-blue-800"
      >
        <span class="font-sans font-black text-white text-lg"
          >현재 접속중인 인원</span
        >
        <div class="flex ml-auto text-white">
          <span class="text-lg animate-pulse mr-1">{{
            onlineUsers.length
          }}</span>
          <svg
            class="h-6 w-6 fill-current"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
            />
          </svg>
        </div>
      </div>
      <div class="flex flex-row items-center justify-between mt-8">
        <span class="cursor-default font-bold text-xl text-gray-900 mr-5"
          >대화방</span
        >
        <div class="flex flex-row items-center justify-between">
          <div @click="createRoom = !createRoom">
            <svg
              class="h-12 text-blue-800 cursor-pointer"
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                stroke-linecap="round"
                stroke-linejoin="round"
                stroke-width="2"
                d="M12 6v6m0 0v6m0-6h6m-6 0H6"
              />
            </svg>
          </div>
        </div>
      </div>
      <div class="flex flex-col relative mt-4">
        <div>
          <input
            v-model="searchkeyword"
            class="rounded-md pl-3 h-10 w-full focus:outline-none bg-gray-100 border border-blue-800 placeholder-gray-500 focus:border-blue-800 focus:bg-white"
            type="text"
            placeholder="방 제목으로 검색해 보세요"
          />
        </div>

        <div class="flex flex-row mt-4">
          <div @click="filterReq = !filterReq">
            <svg
              class="h-8 text-blue-800 cursor-pointer"
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="{2}"
                d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4"
              />
            </svg>
          </div>
          <span
            @click="filterReq = !filterReq"
            class="h-6 cursor-pointer bg-blue-800 mt-1 px-1 flex rounded text-yellow-500"
          >
            MBTI
          </span>
        </div>
        <div
          class="bg-blue-800 rounded-r-md absolute flex items-center justify-center h-10 w-10 right-0 top-0"
        >
          <svg
            class="cursor-defalut h-6 w-6 text-yellow-500 fill-current"
            fill="currentColor"
            viewBox="0 0 20 20"
          >
            <path
              fill-rule="evenodd"
              d="M8 4a4 4 0 100 8 4 4 0 000-8zM2 8a6 6 0 1110.89 3.476l4.817 4.817a1 1 0 01-1.414 1.414l-4.816-4.816A6 6 0 012 8z"
              clip-rule="evenodd"
            />
          </svg>
        </div>
      </div>
      <ul
        class="flex flex-col mt-4 space-y-2 overflow-y-auto"
        style="height: 400px"
      >
        <li
          @click="enterRoom(idx)"
          v-for="(chatroom, idx) in filteredRoomList"
          :key="idx"
          class="flex flex-row items-center relative hover:bg-gray-100 p-2 rounded cursor-pointer"
        >
          <div
            class="absolute flex items-center justify-center right-0 top-0 h-full mr-2 font-medium py-1 px-2 rounded-full"
          >
            <span
              class="flex items-center justify-center text-sm h-3 max-w-full mr-2 p-3 font-normal leading-none flex-initial rounded text-blue-700 bg-blue-100 border border-blue-300"
              >{{ getDist(chatroom.Hostlocation) }}</span
            >
            <span
              class="flex items-center justify-center text-sm h-3 max-w-full mr-2 p-3 font-normal leading-none flex-initial rounded text-blue-700 bg-blue-200 border border-blue-300"
              >{{ chatroom.roomInfo.roomTopic }}</span
            >
            <span
              class="flex items-center justify-center text-sm h-3 max-w-full p-3 font-normal leading-none flex-initial rounded text-blue-700 bg-blue-200 border border-blue-300"
              >{{ chatroom.roomInfo.mbti }}</span
            >
          </div>
          <div class="relative flex-shrink-0">
            <span
              v-if="isOnline(chatroom.roomInfo.host)"
              class="absolute right-0 top-0 border-2 border-white mt-px mr-px flex items-center justify-center h-4 w-4 rounded-full bg-green-500"
            ></span>
            <span class="flex rounded-full w-16 h-16">
              <img
                v-if="chatroom.roomInfo.gender === 'male'"
                src="@/assets/male.svg"
                alt
                class="w-full h-full rounded-full"
              />
              <img
                v-if="chatroom.roomInfo.gender === 'female'"
                src="@/assets/female.svg"
                alt
                class="w-full h-full rounded-full"
              />
            </span>
          </div>
          <div class="flex flex-col ml-4">
            <h3 class="font-bold">{{ chatroom.roomInfo.roomTitle }}</h3>
            <h2 class="text-sm text-gray-600">
              {{ chatroom.roomInfo.roomDesc }}
            </h2>
          </div>
        </li>
      </ul>
      <div
        class="flex flex-row items-center justify-around mt-4 h-16 bg-blue-800"
      >
        <!-- <div class="cursor-pointer flex text-yellow-500 stroke-current">
          <svg class="h-8 w-8" fill="currentColor" viewBox="0 0 20 20">
            <path
              d="M9 6a3 3 0 11-6 0 3 3 0 016 0zM17 6a3 3 0 11-6 0 3 3 0 016 0zM12.93 17c.046-.327.07-.66.07-1a6.97 6.97 0 00-1.5-4.33A5 5 0 0119 16v1h-6.07zM6 11a5 5 0 015 5v1H1v-1a5 5 0 015-5z"
            />
          </svg>
        </div> -->
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
        <!-- <div class="cursor-pointer flex text-yellow-500 stroke-current">
          <svg class="h-8 w-8" fill="currentColor" viewBox="0 0 20 20">
            <path
              fill-rule="evenodd"
              d="M11.49 3.17c-.38-1.56-2.6-1.56-2.98 0a1.532 1.532 0 01-2.286.948c-1.372-.836-2.942.734-2.106 2.106.54.886.061 2.042-.947 2.287-1.561.379-1.561 2.6 0 2.978a1.532 1.532 0 01.947 2.287c-.836 1.372.734 2.942 2.106 2.106a1.532 1.532 0 012.287.947c.379 1.561 2.6 1.561 2.978 0a1.533 1.533 0 012.287-.947c1.372.836 2.942-.734 2.106-2.106a1.533 1.533 0 01.947-2.287c1.561-.379 1.561-2.6 0-2.978a1.532 1.532 0 01-.947-2.287c.836-1.372-.734-2.942-2.106-2.106a1.532 1.532 0 01-2.287-.947zM10 13a3 3 0 100-6 3 3 0 000 6z"
              clip-rule="evenodd"
            />
          </svg>
        </div> -->
      </div>
    </div>
    <div
      v-if="createRoom"
      class="fixed pin overflow-auto w-full h-full bg-smoke-light z-50 flex"
    >
      <div
        class="rounded relative p-8 bg-yellow-100 w-full max-w-md m-auto flex-col flex"
      >
        <FormulateForm v-model="roomInfo">
          <h2 class="text-2xl font-semibold text-gray-900 text-center mb-5">
            대화방 생성
          </h2>
          <svg
            class="w-8 h-8 cursor-pointer absolute right-0 top-0 mt-3 mr-3 text-blue-800"
            @click="createRoom = false"
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
          <FormulateInput
            input-class="mb-3 placeholder-gray-500 bg-gray-100 border border-gray-300 rounded-md px-3 py-2 focus:border-blue-800 outline-none leading-none focus:bg-white w-full"
            type="text"
            name="roomTitle"
            label="제목"
            placeholder="건전한 방 제목을 적어주세요"
            validation="^required|min:2"
          />
          <FormulateInput
            input-class="mb-3 placeholder-gray-500 bg-gray-100 border border-gray-300 rounded-md px-3 py-2 leading-none focus:border-blue-800 focus:bg-white outline-none w-full"
            type="text"
            name="roomDesc"
            label="설명"
            placeholder="간단한 설명을 적어주세요"
            validation="^required|min:2"
          />

          <FormulateInput
            input-class="bg-gray-100 border border-gray-300 rounded-md px-3 py-2 leading-none focus:border-blue-800 focus:bg-white outline-none w-full mb-1"
            type="select"
            name="roomTopic"
            label="주제"
            :options="{
              일상: '일상',
              취미: '취미',
              유머: '유머',
              '수다/채팅': '수다/채팅',
              힐링: '힐링',
              고민: '고민',
              연애상담: '연애상담',
            }"
          />
          <!-- <FormulateInput
            class="text-center bg-blue-800 mt-5 text-white font-medium py-2 rounded-md"
            type="submit"
            label="생성"
          >
          </FormulateInput> -->
          <button
            @click="createChatRoom()"
            class="focus:outline-none w-full text-center bg-blue-800 mt-5 text-white font-medium py-2 rounded-md"
          >
            생성
          </button>
        </FormulateForm>
      </div>
    </div>
    <div
      v-if="DETAILED_CONDITION"
      class="fixed pin overflow-auto w-full h-full bg-smoke-light z-50 flex"
    >
      <div
        class="rounded relative p-8 bg-white w-full max-w-md m-auto flex-col flex"
      >
        <h2 class="text-2xl font-semibold text-gray-700 text-center mb-5">
          상세 조건
        </h2>
        <svg
          class="w-8 h-8 cursor-pointer absolute right-0 top-0 mt-3 mr-3 text-gray-500"
          @click="DETAILED_CONDITION = false"
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
        <FormulateInput
          input-class="border border-gray-400 rounded px-3 py-2 leading-none focus:border-green-500 outline-none w-full mb-1"
          v-model="sort_option"
          :options="{ dist: '거리', mbti: 'MBTI', topic: '주제' }"
          type="select"
          label="정렬 순서"
        />
        <button
          class="text-center bg-blue-500 hover:bg-blue-700 mt-5 text-white font-bold py-2 border border-blue-700 rounded"
        >
          적용하기
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { mapState } from "vuex";
import { yyyyMMddHHmmsss } from "@/utils.js";
import { calculateDistance } from "@/utils.js";
import firebase from "firebase";
// 결제모듈 설정
const IMP = window.IMP;
IMP.init("imp44023513");

export default {
  computed: {
    ...mapState(["uid", "nickname", "mbti", "gender"]),
    isOnline() {
      return (uid) => {
        console.log(uid);
        return this.onlineUsers.includes(uid);
      };
    },
    filteredRoomList() {
      if (!this.roomlist) return;
      console.log();
      return Object.values(this.roomlist).filter((room) => {
        if (this.filterReq) {
          return (
            this.goodMBTI.includes(room.roomInfo.mbti) &&
            (room.roomInfo.roomTitle.includes(this.searchkeyword) ||
              room.roomInfo.roomDesc.includes(this.searchkeyword))
          );
        } else {
          return (
            room.roomInfo.roomTitle.includes(this.searchkeyword) ||
            room.roomInfo.roomDesc.includes(this.searchkeyword)
          );
        }
      });
    },
  },
  created() {
    this.fetchLocation(),
      this.fetchRoomList(),
      this.online(),
      this.getfriends();
  },
  data() {
    return {
      // 방 목록 데이터
      roomlist: null,
      searchkeyword: "",
      filterReq: false,
      goodMBTI: null,
      // 방 생성 관련 data
      onlineUsers: [],
      createRoom: false,
      roomId: null,
      roomInfo: {
        host: null,
        mbti: null,
        gender: null,
        hostnickname: null,
        roomTitle: null,
        roomDesc: null,
        roomTopic: null,
      },
      peerConnection: null,
      location: {
        lat: null,
        lon: null,
      },
      // 친구 목록 관련 data
      friendsList: [],
      friendReqList: [],
      // 검색 관련 data
      DETAILED_CONDITION: false,
      // 알림 관련 data
      isVisible: false,
    };
  },
  mounted() {
    this.filterMBTI();
  },
  methods: {
    filterMBTI() {
      const MBTIIdx = {
        INFP: 0,
        ENFP: 1,
        INFJ: 2,
        ENFJ: 3,
        INTJ: 4,
        ENTJ: 5,
        INTP: 6,
        ENTP: 7,
        ISFP: 8,
        ESFP: 9,
        ISTP: 10,
        ESTP: 11,
        ISFJ: 12,
        ESFJ: 13,
        ISTJ: 14,
        ESTJ: 15,
      };
      const idxMBTI = {
        0: "INFP",
        1: "ENFP",
        2: "INFJ",
        3: "ENFJ",
        4: "INTJ",
        5: "ENTJ",
        6: "INTP",
        7: "ENTP",
        8: "ISFP",
        9: "ESFP",
        10: "ISTP",
        11: "ESTP",
        12: "ISFJ",
        13: "ESFJ",
        14: "ISTJ",
        15: "ESTJ",
      };
      const harmony = [
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
      ];
      const userMBTIIdx = MBTIIdx[this.mbti];
      const good = [];
      harmony[userMBTIIdx].forEach((val, idx) => {
        if (val) {
          good.push(idx);
        }
      });
      const res = [];
      good.forEach((idx) => {
        res.push(idxMBTI[idx]);
      });
      this.goodMBTI = res;
    },
    fetchLocation() {
      if (!("geolocation" in navigator)) {
        return;
      }
      navigator.geolocation.getCurrentPosition((position) => {
        this.location.lat = position.coords.latitude;
        this.location.lon = position.coords.longitude;
        this.$store.commit("location", this.location);
      });
    },
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
    async createChatRoom() {
      // realtimeDB 에 방을 만든다.
      this.roomInfo.host = this.uid;
      this.roomInfo.hostnickname = this.nickname;
      this.roomInfo.mbti = this.mbti;
      this.roomInfo.gender = this.gender;

      this.roomId = "make=" + this.uid + "-time=" + yyyyMMddHHmmsss();
      await this.$db.ref("Rooms/" + this.roomId).set({
        roomInfo: this.roomInfo,
        Hostlocation: this.location,
      });
      this.$router.push({ name: "ChatRoom", params: { id: this.roomId } });
    },
    async fetchRoomList() {
      const roomRef = await this.$db.ref("Rooms/");
      roomRef.on("value", (snapshot) => {
        console.log(snapshot.val());
        this.roomlist = snapshot.val();
        this.applyfilter();
      });
    },
    applyfilter() {
      if (this.sort_option) {
        console.log(this.sort_option);
      }
    },
    enterRoom(idx) {
      const key = Object.keys(this.roomlist)[idx];
      this.$router.push({ name: "ChatRoom", params: { id: key } });
    },
    online() {
      const usersConnectionRef = this.$db.ref("UsersConnection");
      const cUserConnection = (data) => {
        // 온라인인데 온라인 목록에 없는 경우
        if (
          data.val().connection === true &&
          !this.onlineUsers.includes(data.key)
        ) {
          this.onlineUsers.push(data.key);
        }
        // 오프라인인데 온라인 목록에 있는 경우
        else if (
          data.val().connection === false &&
          this.onlineUsers.includes(data.key)
        ) {
          const idx = this.onlineUsers.indexOf(data.key);
          this.onlineUsers.splice(idx, 1);
        }
      };
      usersConnectionRef.on("child_added", (snap) => {
        cUserConnection(snap);
      });
      usersConnectionRef.on("child_changed", (snap) => {
        cUserConnection(snap);
      });
    },
    getDist(loc) {
      if (loc) {
        const { lat, lon } = loc;
        let dist = calculateDistance(
          lat,
          this.location.lat,
          lon,
          this.location.lon
        );
        dist = Math.round(dist / 1000);
        return dist + "km";
      } else {
        return "거리정보없음";
      }
    },
    async getfriends() {
      const friendsRef = await this.$firestore
        .collection("users")
        .doc(this.uid);
      friendsRef.onSnapshot((snapshot) => {
        const friendsdata = snapshot.data()["friends"];
        const friendsReqdata = snapshot.data()["friendsReq"];
        if (friendsdata) {
          friendsdata.forEach((fri) => {
            if (!this.friendsList.includes(fri)) {
              this.friendsList.push(fri);
            }
          });
        }
        if (friendsReqdata) {
          friendsReqdata.forEach((fri) => {
            if (!this.friendReqList.includes(fri)) {
              this.friendReqList.push(fri);
            }
          });
        }
      });
    },
    async rejectFriend(fri, idx) {
      this.friendReqList.splice(idx, 1);
      const friendReqRef = await this.$firestore
        .collection("users")
        .doc(this.uid);
      await friendReqRef.update({
        friendsReq: this.$firebase.firestore.FieldValue.arrayRemove(fri),
      });
    },
    async acceptFriend(fri, idx) {
      this.friendReqList.splice(idx, 1);
      const friendReqRef = await this.$firestore
        .collection("users")
        .doc(this.uid);
      await friendReqRef.update({
        friendsReq: this.$firebase.firestore.FieldValue.arrayRemove(fri),
      });
      await friendReqRef.update({
        friends: this.$firebase.firestore.FieldValue.arrayUnion(fri),
      });

      const REQUESTERREF = await this.$firestore
        .collection("users")
        .doc(fri.uid);
      await REQUESTERREF.update({
        friends: this.$firebase.firestore.FieldValue.arrayUnion({
          uid: this.uid,
          nickname: this.nickname,
        }),
      });
    },
    requestPay() {
      IMP.request_pay(
        {
          pg: "kakaopay",
          pay_method: "card",
          merchant_uid: "merchant_" + this.uid + new Date().getTime(),
          name: "주문명 : 업굴 결제(테스트)",
          amount: 1000,
          buyer_name: this.uid,
        },
        (rsp) => {
          if (rsp.success) {
            console.log("성공");
          } else {
            console.log(rsp);
            console.log("실패");
          }
        }
      );
    },
  },
};
</script>

<style></style>
