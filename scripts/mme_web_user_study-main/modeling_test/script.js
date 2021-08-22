window.addEventListener('load',()=>{
    const params = (new URL(document.location)).searchParams;
    let chosen_condition = params.get('condition');
    const condition_url = "/mme_web_user_study/group".concat(chosen_condition).concat("/index.html");
    document.getElementById("form_only").action=condition_url;
})


