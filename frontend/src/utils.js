// eslint-disable-next-line no-unused-vars

export const yyyyMMddHHmmsss = () => {
    let vDate = new Date();
    let yyyy = vDate.getFullYear().toString();
    let MM = (vDate.getMonth() + 1).toString();
    let dd = vDate.getDate().toString();
    let HH = vDate.getHours().toString();
    let mm = vDate.getMinutes().toString();
    let ss = vDate.getSeconds().toString();
    let sss = vDate.getMilliseconds().toString();
    return yyyy + (MM[1] ? MM : '0'+MM[0]) + (dd[1] ? dd : '0'+dd[0]) + (HH[1] ? HH : '0'+ HH[0]) + (mm[1] ? mm : '0'+ mm[0]) + (ss[1] ? ss : '0'+ss[0])+ sss;
}

export const parseRoomLink = (roomLink) => {
    const lidx = roomLink.indexOf('-');
    return roomLink.substring(5, lidx);
}

export const pad = (n) => {
    return n > 9 ? "" + n : "0" + n;
}

export const timestamptotime = (timestamp) => {
    const date = new Date(timestamp);
    const year = date.getFullYear();
    const month = date.getMonth() + 1;
    const day = date.getDate();
    const hour = date.getHours();
    const minute = date.getMinutes();
    const week = new Array("일", "월", "화", "수", "목", "금", "토");
    const convertDate = year + "년 "+month+"월 "+ day +"일 ("+ week[date.getDay()] +") ";
    let convertHour="";
    if(hour < 12){
        convertHour = "오전 " + pad(hour) +":" + pad(minute);
    }else if(hour === 12){
        convertHour = "오후 " + pad(hour) +":" + pad(minute);
    }else{
        convertHour = "오후 " + pad(hour - 12) +":" + pad(minute);
    }
    return convertDate + convertHour;
};

export const calculateDistance = (lat1,lat2, lon1, lon2) => {
    const R = 6371e3; // metres
    const φ1 = lat1 * Math.PI/180; // φ, λ in radians
    const φ2 = lat2 * Math.PI/180;
    const Δφ = (lat2-lat1) * Math.PI/180;
    const Δλ = (lon2-lon1) * Math.PI/180;

    const a = Math.sin(Δφ/2) * Math.sin(Δφ/2) +
        Math.cos(φ1) * Math.cos(φ2) *
        Math.sin(Δλ/2) * Math.sin(Δλ/2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));

    const d = R * c; // in metres
    return d;
}
