package com.example.automooden

import android.Manifest
import android.os.Bundle
import android.widget.Toast

import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent

import androidx.activity.result.contract.ActivityResultContracts

import androidx.compose.foundation.layout.*

import androidx.compose.material3.*

import androidx.compose.runtime.*

import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

import androidx.compose.ui.platform.LocalContext

import kotlinx.coroutines.*

class MainActivity : ComponentActivity() {

    private val cameraPermissionLauncher =

        registerForActivityResult(
            ActivityResultContracts.RequestPermission()
        ) { granted ->

            if (!granted) {

                Toast.makeText(

                    this,

                    "Камера недоступна",

                    Toast.LENGTH_LONG

                ).show()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {

        super.onCreate(savedInstanceState)

        NotificationHelper.createChannel(this)

        cameraPermissionLauncher.launch(
            Manifest.permission.CAMERA
        )

        setContent {

            MaterialTheme {

                MainScreen()
            }
        }
    }
}

@Composable
fun MainScreen() {

    val context = LocalContext.current

    var plateNumber by remember {

        mutableStateOf("")
    }

    var result by remember {

        mutableStateOf<CarResponse?>(null)
    }

    var loading by remember {

        mutableStateOf(false)
    }

    Column(

        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),

        horizontalAlignment =
            Alignment.CenterHorizontally

    ) {

        Spacer(
            modifier = Modifier.height(40.dp)
        )

        Text(

            text = "AutoMooden",

            style =
                MaterialTheme.typography.headlineLarge
        )

        Spacer(
            modifier = Modifier.height(30.dp)
        )

        OutlinedTextField(

            value = plateNumber,

            onValueChange = {

                plateNumber = it.uppercase()
            },

            label = {

                Text("Введите номер")
            },

            modifier = Modifier.fillMaxWidth()
        )

        Spacer(
            modifier = Modifier.height(20.dp)
        )

        Button(

            onClick = {

                if (plateNumber.isBlank()) {

                    Toast.makeText(

                        context,

                        "Введите номер",

                        Toast.LENGTH_SHORT

                    ).show()

                    return@Button
                }

                loading = true

                CoroutineScope(
                    Dispatchers.IO
                ).launch {

                    try {

                        val response =
                            RetrofitClient
                                .api
                                .getCar(plateNumber)

                        withContext(
                            Dispatchers.Main
                        ) {

                            result = response

                            loading = false

                            if (
                                response.found &&
                                response.rating < 3.0
                            ) {

                                NotificationHelper
                                    .showDangerNotification(

                                        context,

                                        response.plate_number
                                    )
                            }
                        }

                    } catch (e: Exception) {

                        e.printStackTrace()

                        withContext(
                            Dispatchers.Main
                        ) {

                            loading = false

                            Toast.makeText(

                                context,

                                "Ошибка подключения",

                                Toast.LENGTH_LONG

                            ).show()
                        }
                    }
                }
            },

            modifier = Modifier.fillMaxWidth()
        ) {

            Text("Проверить")
        }

        Spacer(
            modifier = Modifier.height(30.dp)
        )

        if (loading) {

            CircularProgressIndicator()
        }

        result?.let { car ->

            Card(

                modifier = Modifier.fillMaxWidth()

            ) {

                Column(

                    modifier =
                        Modifier.padding(20.dp)

                ) {

                    Text(
                        text =
                            "Номер: ${car.plate_number}"
                    )

                    Spacer(
                        modifier =
                            Modifier.height(8.dp)
                    )

                    Text(
                        text =
                            "Владелец: ${car.owner_name}"
                    )

                    Spacer(
                        modifier =
                            Modifier.height(8.dp)
                    )

                    Text(
                        text =
                            "Рейтинг: ${car.rating}"
                    )

                    Spacer(
                        modifier =
                            Modifier.height(8.dp)
                    )

                    Text(
                        text =
                            "Голосов: ${car.votes}"
                    )

                    Spacer(
                        modifier =
                            Modifier.height(8.dp)
                    )

                    Text(
                        text =
                            "Предупреждение: ${car.warning}"
                    )

                    if (
                        car.rating < 3.0
                    ) {

                        Spacer(
                            modifier =
                                Modifier.height(16.dp)
                        )

                        Text(
                            text =
                                "⚠ Опасный водитель"
                        )
                    }
                }
            }
        }
    }
}