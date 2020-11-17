<template>
  <div id="app">
    <router-view />
  </div>
</template>
<style>
</style>
<script>
import firebase from "firebase";

export default {
  methods: {
    // 로그인한 상태를 확인함
    async getFirebaseUser() {
      await this.$firebase.auth().onAuthStateChanged((user) => {
        if (user) {
          this.checkOnline(user.uid);
          console.log("로그인 되어있음");
        } else if (this.$router.history.current.name !== 'Signup') {                  
          this.$router.push({name : "Login"});
        }
      });
    },
    checkOnline(uid) {
      const userUid = uid;
      const myConnectionRef = this.$db.ref(
        "UsersConnection/" + userUid + "/connection"
      );
      const lastOnlineRef = this.$db.ref(
        "UsersConnection/" + userUid + "/lastOnline"
      );
      const connectedRef = this.$db.ref(".info/connected");
      connectedRef.on("value", (snap) => {
        if (snap.val() === true) {
          myConnectionRef.set(true);

          myConnectionRef.onDisconnect().set(false);
          lastOnlineRef
            .onDisconnect()
            .set(firebase.database.ServerValue.TIMESTAMP);
        }
      });
    },
  },
  async created() {
    await this.getFirebaseUser();
  },
};
</script>
