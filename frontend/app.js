const config = window.CINEMA_ATLAS_CONFIG || {};
const apiBaseUrl = (config.API_BASE_URL || "").replace(/\/$/, "");
const isLocal = window.location.hostname === "127.0.0.1" || window.location.hostname === "localhost";
const apiBase = apiBaseUrl && !apiBaseUrl.includes("your-render-backend")
    ? apiBaseUrl
    : (isLocal ? "http://127.0.0.1:5000" : "");

const state = {
    genre: "all",
    language: "all",
};

const hero = document.getElementById("hero");
const heroTitle = document.getElementById("hero-title");
const heroMeta = document.getElementById("hero-meta");
const heroOverview = document.getElementById("hero-overview");
const genreSelect = document.getElementById("genre");
const languageSelect = document.getElementById("language");
const searchForm = document.getElementById("search-form");
const titleInput = document.getElementById("title");
const suggestionsBox = document.getElementById("suggestions");
const messageSection = document.getElementById("message-section");
const messageText = document.getElementById("message-text");
const recommendationSection = document.getElementById("recommendation-section");
const recommendationHeading = document.getElementById("recommendation-heading");
const recommendationGrid = document.getElementById("recommendation-grid");
const popularGrid = document.getElementById("popular-grid");
const seedCard = document.getElementById("seed-card");
const seedPoster = document.getElementById("seed-poster");
const seedTitle = document.getElementById("seed-title");
const seedMeta = document.getElementById("seed-meta");
const seedOverview = document.getElementById("seed-overview");

function buildUrl(path, params = {}) {
    const url = new URL(`${apiBase}${path}`);
    Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined && value !== null && value !== "") {
            url.searchParams.set(key, value);
        }
    });
    return url.toString();
}

async function fetchJson(path, params = {}) {
    if (!apiBase) {
        throw new Error("Set CINEMA_ATLAS_API_BASE_URL for Vercel builds or update frontend/config.js for local testing.");
    }

    const response = await fetch(buildUrl(path, params));
    const payload = await response.json();
    if (!response.ok) {
        throw new Error(payload.error || "Request failed.");
    }
    return payload;
}

function renderHero(movie) {
    if (!movie) {
        return;
    }
    heroTitle.textContent = movie.title;
    heroOverview.textContent = movie.overview;
    heroMeta.innerHTML = [
        `<span>IMDb ${movie.imdb_rating}</span>`,
        `<span>${movie.genres}</span>`,
        `<span>${movie.language}</span>`
    ].join("");
    if (movie.backdrop_url) {
        hero.style.backgroundImage = `linear-gradient(180deg, rgba(7, 10, 18, 0.18), rgba(7, 10, 18, 0.96)), url('${movie.backdrop_url}')`;
    }
}

function movieCard(movie) {
    const poster = movie.poster_url
        ? `<img src="${movie.poster_url}" alt="${movie.title} poster">`
        : `<div class="poster-fallback">${movie.title}</div>`;
    return `
        <article class="movie-card" data-movie-id="${movie.id}">
            <div class="poster-frame">
                ${poster}
                <span class="rating-chip">IMDb ${movie.imdb_rating}</span>
            </div>
            <div class="movie-copy">
                <h4>${movie.title}</h4>
                <p>${movie.year || "n/a"} | ${movie.genres}</p>
                <span>${movie.language}</span>
            </div>
        </article>
    `;
}

function attachMovieClicks(container) {
    container.querySelectorAll("[data-movie-id]").forEach((card) => {
        card.addEventListener("click", () => {
            const movieId = card.getAttribute("data-movie-id");
            loadRecommendations({ movie_id: movieId });
        });
    });
}

function renderMovieGrid(container, movies) {
    container.innerHTML = movies.map(movieCard).join("");
    attachMovieClicks(container);
}

function renderSeedMovie(movie) {
    if (!movie) {
        seedCard.classList.add("hidden");
        return;
    }
    seedCard.classList.remove("hidden");
    seedPoster.innerHTML = movie.poster_url
        ? `<img src="${movie.poster_url}" alt="${movie.title} poster">`
        : `<div class="poster-fallback">${movie.title}</div>`;
    seedTitle.textContent = movie.title;
    seedMeta.textContent = `${movie.year || "n/a"} | ${movie.genres} | ${movie.runtime}`;
    seedOverview.textContent = movie.overview;
}

function showMessage(message, isError = false) {
    if (!message) {
        messageSection.classList.add("hidden");
        messageText.textContent = "";
        messageText.style.color = "";
        return;
    }
    messageSection.classList.remove("hidden");
    messageText.textContent = message;
    messageText.style.color = isError ? "#ffb6b6" : "";
}

async function loadFilters() {
    const payload = await fetchJson("/api/filters");
    genreSelect.innerHTML = payload.genres
        .map((genre) => `<option value="${genre}">${genre === "all" ? "All genres" : genre}</option>`)
        .join("");
    languageSelect.innerHTML = payload.languages
        .map((entry) => `<option value="${entry.code}">${entry.label}</option>`)
        .join("");
}

async function loadFeatured() {
    const payload = await fetchJson("/api/featured");
    renderHero(payload.movie);
}

async function loadPopular() {
    const payload = await fetchJson("/api/popular", {
        genre: state.genre,
        language: state.language,
        limit: 18,
    });
    renderMovieGrid(popularGrid, payload.movies);
}

async function loadRecommendations(params) {
    try {
        const payload = await fetchJson("/api/recommendations", {
            ...params,
            genre: state.genre,
            language: state.language,
            limit: 18,
        });
        const result = payload.result;
        recommendationHeading.textContent = `Because you searched for ${result.resolved_title}`;
        renderSeedMovie(result.seed_movie);
        renderMovieGrid(recommendationGrid, result.movies);
        recommendationSection.classList.remove("hidden");
        showMessage(payload.user_message || result.message || "");
        recommendationSection.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (error) {
        recommendationSection.classList.add("hidden");
        showMessage(error.message, true);
    }
}

let suggestionTimeout;
titleInput.addEventListener("input", () => {
    clearTimeout(suggestionTimeout);
    const query = titleInput.value.trim();
    if (query.length < 2) {
        suggestionsBox.hidden = true;
        suggestionsBox.innerHTML = "";
        return;
    }
    suggestionTimeout = setTimeout(async () => {
        try {
            const payload = await fetchJson("/api/suggestions", { q: query });
            const suggestions = payload.suggestions || [];
            if (!suggestions.length) {
                suggestionsBox.hidden = true;
                suggestionsBox.innerHTML = "";
                return;
            }
            suggestionsBox.hidden = false;
            suggestionsBox.innerHTML = suggestions
                .map((movie) => `<div class="suggestion-item" data-title="${movie.title}">${movie.title}</div>`)
                .join("");

            suggestionsBox.querySelectorAll(".suggestion-item").forEach((node) => {
                node.addEventListener("click", () => {
                    titleInput.value = node.getAttribute("data-title");
                    suggestionsBox.hidden = true;
                });
            });
        } catch (error) {
            suggestionsBox.hidden = true;
            suggestionsBox.innerHTML = "";
        }
    }, 180);
});

document.addEventListener("click", (event) => {
    if (!event.target.closest(".search-shell")) {
        suggestionsBox.hidden = true;
    }
});

genreSelect.addEventListener("change", async () => {
    state.genre = genreSelect.value;
    await loadPopular();
});

languageSelect.addEventListener("change", async () => {
    state.language = languageSelect.value;
    await loadPopular();
});

searchForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const title = titleInput.value.trim();
    if (!title) {
        showMessage("Enter a movie title to search.", true);
        return;
    }
    await loadRecommendations({ title });
});

async function bootstrap() {
    try {
        await Promise.all([loadFilters(), loadFeatured()]);
        state.genre = genreSelect.value || "all";
        state.language = languageSelect.value || "all";
        await loadPopular();
        showMessage("");
    } catch (error) {
        showMessage(error.message, true);
    }
}

bootstrap();
