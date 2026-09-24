let activeEditPersonId = null;

document.addEventListener('DOMContentLoaded', () => {
    loadProfiles();
    loadEvents();
    
    // Auto-refresh gallery and activity log every 3 seconds
    setInterval(loadProfiles, 3000);
    setInterval(loadEvents, 2500);
});

async function loadProfiles() {
    try {
        const response = await fetch('/api/profiles');
        const data = await response.json();
        
        if (data.success) {
            renderGallery(data.profiles);
            document.getElementById('statProfilesCount').innerText = data.total;
        }
    } catch (err) {
        console.error('Error fetching profiles:', err);
    }
}

function renderGallery(profiles) {
    const galleryContainer = document.getElementById('profilesGallery');
    
    if (!profiles || profiles.length === 0) {
        galleryContainer.innerHTML = `
            <div class="empty-state">
                <i class="fa-solid fa-user-clock"></i>
                <p>No identities detected yet. Step in front of the camera to enroll!</p>
            </div>
        `;
        return;
    }
    
    galleryContainer.innerHTML = profiles.map(profile => {
        const thumbUrl = profile.thumbnail ? `${profile.thumbnail}?t=${new Date().getTime()}` : '';
        const thumbHtml = thumbUrl 
            ? `<img src="${thumbUrl}" alt="${profile.name}" />` 
            : `<i class="fa-solid fa-user" style="font-size: 42px; color: #475569;"></i>`;
            
        return `
            <div class="profile-card">
                <div class="profile-thumb-box">
                    ${thumbHtml}
                </div>
                <div class="profile-info">
                    <span class="profile-name">${escapeHtml(profile.name)}</span>
                    <span class="profile-id">${profile.id}</span>
                    <span class="profile-meta"><i class="fa-solid fa-eye"></i> ${profile.sightings} sightings</span>
                    <span class="profile-meta"><i class="fa-solid fa-clock"></i> Last seen: ${profile.last_seen || 'Just now'}</span>
                </div>
                <div class="profile-actions">
                    <button class="btn btn-sm btn-secondary" onclick="openEditModal('${profile.id}', '${escapeHtml(profile.name)}')">
                        <i class="fa-solid fa-pen"></i> Rename
                    </button>
                    <button class="btn btn-sm btn-danger-outline" onclick="deleteProfile('${profile.id}')">
                        <i class="fa-solid fa-trash"></i>
                    </button>
                </div>
            </div>
        `;
    }).join('');
}

async function loadEvents() {
    try {
        const response = await fetch('/api/events');
        const data = await response.json();
        const eventsLog = document.getElementById('eventsLog');
        
        if (data.events && data.events.length > 0) {
            eventsLog.innerHTML = data.events.slice(0, 15).map(evt => `
                <div class="event-item">
                    <i class="fa-solid fa-user-plus text-accent"></i>
                    <div>
                        <strong>${escapeHtml(evt.title)} (${evt.time})</strong>
                        <p>${escapeHtml(evt.details)}</p>
                    </div>
                </div>
            `).join('');
        }
    } catch (err) {
        console.error('Error fetching events:', err);
    }
}

function openEditModal(personId, currentName) {
    activeEditPersonId = personId;
    document.getElementById('modalPersonId').innerText = personId;
    document.getElementById('modalInputName').value = currentName;
    document.getElementById('editModal').classList.add('show');
}

function closeModal() {
    activeEditPersonId = null;
    document.getElementById('editModal').classList.remove('show');
}

async function saveName() {
    const newName = document.getElementById('modalInputName').value.trim();
    if (!activeEditPersonId || !newName) return;
    
    try {
        const response = await fetch('/api/rename', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ person_id: activeEditPersonId, name: newName })
        });
        const res = await response.json();
        if (res.success) {
            closeModal();
            loadProfiles();
        } else {
            alert('Failed to rename: ' + res.error);
        }
    } catch (err) {
        console.error('Error saving name:', err);
    }
}

async function deleteProfile(personId) {
    if (!confirm(`Are you sure you want to delete profile ${personId}?`)) return;
    
    try {
        const response = await fetch('/api/delete', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ person_id: personId })
        });
        const res = await response.json();
        if (res.success) {
            loadProfiles();
        }
    } catch (err) {
        console.error('Error deleting profile:', err);
    }
}

async function clearDatabase() {
    if (!confirm('Are you sure you want to reset ALL registered identities?')) return;
    
    try {
        const response = await fetch('/api/clear', { method: 'POST' });
        const res = await response.json();
        if (res.success) {
            loadProfiles();
        }
    } catch (err) {
        console.error('Error resetting database:', err);
    }
}

function escapeHtml(str) {
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}
