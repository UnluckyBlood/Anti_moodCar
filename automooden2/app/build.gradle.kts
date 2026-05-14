plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.example.automooden"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.example.automooden"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner =
            "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        compose = true
    }

    composeOptions {
        kotlinCompilerExtensionVersion = "1.5.14"
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")

    implementation(
        "androidx.lifecycle:lifecycle-runtime-ktx:2.8.0"
    )

    implementation(
        "androidx.activity:activity-compose:1.9.0"
    )

    implementation(
        platform("androidx.compose:compose-bom:2024.05.00")
    )

    implementation("androidx.compose.ui:ui")

    implementation("androidx.compose.ui:ui-graphics")

    implementation(
        "androidx.compose.ui:ui-tooling-preview"
    )

    implementation("androidx.compose.material3:material3")

    // CAMERA X
    implementation("androidx.camera:camera-core:1.3.3")

    implementation("androidx.camera:camera-camera2:1.3.3")

    implementation(
        "androidx.camera:camera-lifecycle:1.3.3"
    )

    implementation("androidx.camera:camera-view:1.3.3")

    // ML KIT
    implementation(
        "com.google.mlkit:text-recognition:16.0.0"
    )

    // RETROFIT
    implementation("com.squareup.retrofit2:retrofit:2.11.0")

    implementation(
        "com.squareup.retrofit2:converter-gson:2.11.0"
    )

    // TOOLING
    debugImplementation(
        "androidx.compose.ui:ui-tooling"
    )
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.appcompat:appcompat:1.6.1")
}