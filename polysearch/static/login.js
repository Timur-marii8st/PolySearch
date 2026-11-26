"use strict";
const form = document.getElementById("login-form");
form.addEventListener("submit", async e => {
  e.preventDefault();
  const button = form.querySelector("button"), error = document.getElementById("login-error");
  button.disabled = true; error.hidden = true;
  try {
    const response = await fetch("/api/auth/login", {method:"POST",
      headers:{"Content-Type":"application/json","X-PolySearch-Request":"1"},
      body:JSON.stringify(Object.fromEntries(new FormData(form)))});
    if (!response.ok) throw Error(response.status===429 ? "Слишком много попыток. Повторите позднее." : "Не удалось войти. Проверьте данные.");
    location.replace("/");
  } catch (e) { error.textContent=e.message; error.hidden=false; }
  finally { form.elements.password.value=""; button.disabled=false; }
});
